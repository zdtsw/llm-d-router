/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package scheduling

import (
	"context"
	"reflect"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common/observability/semconv"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	schedprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/prefix"
)

// setupPickerSpanRecorder installs an in-memory span recorder as the global
// tracer provider and returns it, restoring the previous provider on cleanup.
func setupPickerSpanRecorder(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	origTP := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { otel.SetTracerProvider(origTP) })
	return recorder
}

func findPickerSpans(spans []sdktrace.ReadOnlySpan, name string) []sdktrace.ReadOnlySpan {
	var out []sdktrace.ReadOnlySpan
	for _, s := range spans {
		if s.Name() == name {
			out = append(out, s)
		}
	}
	return out
}

func pickerSpanAttributes(span sdktrace.ReadOnlySpan) map[attribute.Key]attribute.Value {
	attrs := make(map[attribute.Key]attribute.Value)
	for _, kv := range span.Attributes() {
		attrs[kv.Key] = kv.Value
	}
	return attrs
}

func newWeightedScores(names ...string) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(names))
	for i, name := range names {
		endpoint := fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}}, nil, nil)
		scores[endpoint] = float64(i)
	}
	return scores
}

// fakePicker returns the first selected candidates as targets, or a nil result
// when nilResult is set, and optionally emits a child span from its context.
type fakePicker struct {
	typedName fwkplugin.TypedName
	selected  int
	nilResult bool
	emitChild bool
}

var _ fwksched.Picker = &fakePicker{}

func (f *fakePicker) TypedName() fwkplugin.TypedName { return f.typedName }

func (f *fakePicker) Pick(ctx context.Context, scoredPods []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	if f.emitChild {
		_, span := otel.Tracer("test").Start(ctx, "inner_picker_span")
		span.End()
	}
	if f.nilResult {
		return nil
	}
	targets := make([]fwksched.Endpoint, 0, f.selected)
	for i := 0; i < f.selected && i < len(scoredPods); i++ {
		targets = append(targets, scoredPods[i].Endpoint)
	}
	return &fwksched.ProfileRunResult{TargetEndpoints: targets}
}

func TestRunPickerPluginSingleSpan(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "max-score", Name: "instance-a"}, selected: 1}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1", "pod2", "pod3")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{TargetModel: "m1", Headers: map[string]string{reqcommon.RequestIDHeaderKey: "r1"}}, scores)
	root.End()

	if result == nil || len(result.TargetEndpoints) != 1 {
		t.Fatalf("runPickerPlugin returned %v, want 1 target", result)
	}

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	span := spans[0]
	if span.SpanKind() != trace.SpanKindInternal {
		t.Errorf("span kind = %v, want Internal", span.SpanKind())
	}
	if span.Parent().SpanID() != root.SpanContext().SpanID() {
		t.Errorf("parent span ID = %v, want root %v", span.Parent().SpanID(), root.SpanContext().SpanID())
	}

	attrs := pickerSpanAttributes(span)
	for _, tc := range []struct {
		key  attribute.Key
		want string
	}{
		{semconv.GenAIRequestModelKey, "m1"},
		{semconv.GenAIRequestIDKey, "r1"},
	} {
		if got := attrs[tc.key].AsString(); got != tc.want {
			t.Errorf("%s = %q, want %q", tc.key, got, tc.want)
		}
	}
	if got := attrs[semconv.LLMDEPPPickerCandidateEndpointsKey].AsInt64(); got != 3 {
		t.Errorf("candidate_endpoints = %d, want 3", got)
	}
	// newWeightedScores assigns score i to the i-th name, so the span records
	// candidates highest score first.
	if got, want := attrs[semconv.LLMDEPPPickerTopEndpointsKey].AsStringSlice(), []string{"/pod3", "/pod2", "/pod1"}; !reflect.DeepEqual(got, want) {
		t.Errorf("top_endpoints = %v, want %v", got, want)
	}
	if got, want := attrs[semconv.LLMDEPPPickerTopScoresKey].AsFloat64Slice(), []float64{2, 1, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("top_scores = %v, want %v", got, want)
	}
}

func TestRunPickerPluginTopScoresCapped(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "max-score", Name: "multi"}, selected: 2}
	profile := NewSchedulerProfile().WithPicker(picker)
	// Seven candidates (scores 0..6) exceed the cap of five.
	scores := newWeightedScores("pod1", "pod2", "pod3", "pod4", "pod5", "pod6", "pod7")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	attrs := pickerSpanAttributes(spans[0])
	if got, want := attrs[semconv.LLMDEPPPickerTopEndpointsKey].AsStringSlice(), []string{"/pod7", "/pod6", "/pod5", "/pod4", "/pod3"}; !reflect.DeepEqual(got, want) {
		t.Errorf("top_endpoints = %v, want the 5 highest-scoring %v", got, want)
	}
	if got, want := attrs[semconv.LLMDEPPPickerTopScoresKey].AsFloat64Slice(), []float64{6, 5, 4, 3, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("top_scores = %v, want %v", got, want)
	}
}

func TestRunPickerPluginNilResult(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "noop-picker", Name: "noop"}, nilResult: true}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1", "pod2")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	result := profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	if result != nil {
		t.Fatalf("runPickerPlugin returned %v, want nil", result)
	}
	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1 (span must end on nil result)", len(spans))
	}
	// Scores are captured from the candidates before Pick, so they are recorded
	// even when the picker selects nothing.
	if got, want := pickerSpanAttributes(spans[0])[semconv.LLMDEPPPickerTopScoresKey].AsFloat64Slice(), []float64{1, 0}; !reflect.DeepEqual(got, want) {
		t.Errorf("top_scores = %v, want %v", got, want)
	}
}

func TestRunPickerPluginNestsDelegateSpan(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "child", Name: "c"}, selected: 1, emitChild: true}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	outer := findPickerSpans(recorder.Ended(), "pick_endpoints")
	inner := findPickerSpans(recorder.Ended(), "inner_picker_span")
	if len(outer) != 1 || len(inner) != 1 {
		t.Fatalf("got %d pick_endpoints and %d inner spans, want 1 each", len(outer), len(inner))
	}
	if inner[0].Parent().SpanID() != outer[0].SpanContext().SpanID() {
		t.Errorf("inner span parent = %v, want pick_endpoints span %v",
			inner[0].Parent().SpanID(), outer[0].SpanContext().SpanID())
	}
}

func TestRunPickerPluginOmitsEmptyGenAI(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	picker := &fakePicker{typedName: fwkplugin.TypedName{Type: "p", Name: "n"}, selected: 1}
	profile := NewSchedulerProfile().WithPicker(picker)
	scores := newWeightedScores("pod1")

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	profile.runPickerPlugin(ctx, &fwksched.InferenceRequest{}, scores)
	root.End()

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	attrs := pickerSpanAttributes(spans[0])
	if _, ok := attrs[semconv.GenAIRequestModelKey]; ok {
		t.Errorf("%s set for empty TargetModel", semconv.GenAIRequestModelKey)
	}
	if _, ok := attrs[semconv.GenAIRequestIDKey]; ok {
		t.Errorf("%s set without x-request-id", semconv.GenAIRequestIDKey)
	}
}

func TestRunPickerPluginPrefixCacheTierAttribution(t *testing.T) {
	recorder := setupPickerSpanRecorder(t)

	const producerName = "precise-prefix-cache-producer"
	key := attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName)
	selected := fwkdl.NewAttributes()
	selected.Put(key, attrprefix.NewPrefixCacheMatchInfo(4, 10, 16).
		WithCachedBlocksByTier(map[string]int{"gpu": 3, "cpu": 1}).
		WithTierWeights(map[string]float64{"gpu": 1, "cpu": 0.8}))
	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod2"}}, nil, selected),
	}
	scorer, err := schedprefix.New(context.Background(), "", producerName)
	if err != nil {
		t.Fatalf("New prefix-cache-scorer: %v", err)
	}
	picker := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-picker", Name: "picker"},
		PickRes:   k8stypes.NamespacedName{Name: "pod2"},
	}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorer, 0.5)).
		WithPicker(picker)

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	_, err = profile.Run(ctx, &fwksched.InferenceRequest{TargetModel: "m1"}, endpoints)
	root.End()
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	attrs := pickerSpanAttributes(spans[0])
	if got := attrs[semconv.LLMDEPPPickerSelectedEndpointKey].AsString(); got != "/pod2" {
		t.Errorf("selected_endpoint = %q, want %q", got, "/pod2")
	}
	if got := attrs[semconv.LLMDEPPPickerPrefixCacheTiersKey].AsStringSlice(); !reflect.DeepEqual(got, []string{"cpu", "gpu"}) {
		t.Errorf("prefix_cache.tiers = %v, want [cpu gpu]", got)
	}
	if got := attrs[semconv.LLMDEPPPickerPrefixCacheTierCachedBlocksKey].AsFloat64Slice(); !reflect.DeepEqual(got, []float64{1, 3}) {
		t.Errorf("prefix_cache.tier.cached_blocks = %v, want [1 3]", got)
	}
	// The values are the tier weights from the match info. The scorer weight
	// in this profile is 0.5 and must not appear here.
	if got := attrs[semconv.LLMDEPPPickerPrefixCacheTierWeightsKey].AsFloat64Slice(); !reflect.DeepEqual(got, []float64{0.8, 1}) {
		t.Errorf("prefix_cache.tier.weights = %v, want [0.8 1]", got)
	}
}

// pickWithSelectedMatchInfo runs a one-endpoint profile whose only endpoint
// carries info under the prefix scorer's data key, and returns the attributes of
// the pick_endpoints span. A nil scorer selects a prefix-cache scorer; a
// non-nil scorer replaces it.
func pickWithSelectedMatchInfo(t *testing.T, info *attrprefix.PrefixCacheMatchInfo, scorer fwksched.Scorer) map[attribute.Key]attribute.Value {
	t.Helper()
	recorder := setupPickerSpanRecorder(t)

	const producerName = "precise-prefix-cache-producer"
	attrs := fwkdl.NewAttributes()
	attrs.Put(attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(producerName), info)
	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "pod1"}}, nil, attrs),
	}
	if scorer == nil {
		prefixScorer, err := schedprefix.New(context.Background(), "", producerName)
		if err != nil {
			t.Fatalf("New prefix-cache-scorer: %v", err)
		}
		scorer = prefixScorer
	}
	picker := &testPlugin{
		typedName: fwkplugin.TypedName{Type: "test-picker", Name: "picker"},
		PickRes:   k8stypes.NamespacedName{Name: "pod1"},
	}
	profile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(scorer, 1)).
		WithPicker(picker)

	ctx, root := otel.Tracer("test").Start(context.Background(), "root")
	_, err := profile.Run(ctx, &fwksched.InferenceRequest{TargetModel: "m1"}, endpoints)
	root.End()
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}

	spans := findPickerSpans(recorder.Ended(), "pick_endpoints")
	if len(spans) != 1 {
		t.Fatalf("got %d pick_endpoints spans, want 1", len(spans))
	}
	return pickerSpanAttributes(spans[0])
}

var prefixCacheAttributionKeys = []attribute.Key{
	semconv.LLMDEPPPickerSelectedEndpointKey,
	semconv.LLMDEPPPickerPrefixCacheTiersKey,
	semconv.LLMDEPPPickerPrefixCacheTierCachedBlocksKey,
	semconv.LLMDEPPPickerPrefixCacheTierWeightsKey,
}

func TestRunPickerPluginOmitsPrefixCacheAttribution(t *testing.T) {
	tests := []struct {
		name   string
		info   *attrprefix.PrefixCacheMatchInfo
		scorer fwksched.Scorer
	}{
		{
			name: "no tier data",
			info: attrprefix.NewPrefixCacheMatchInfo(4, 10, 16),
		},
		{
			name: "empty tier data",
			info: attrprefix.NewPrefixCacheMatchInfo(0, 10, 16).
				WithCachedBlocksByTier(map[string]int{}).
				WithTierWeights(map[string]float64{"gpu": 1}),
		},
		{
			name: "only zero-block tiers",
			info: attrprefix.NewPrefixCacheMatchInfo(0, 10, 16).
				WithCachedBlocksByTier(map[string]int{"gpu": 0}).
				WithTierWeights(map[string]float64{"gpu": 1}),
		},
		{
			// The producer publishes tier data to every profile. A profile that
			// does not score on it must not report it as a routing influence.
			name: "profile without a prefix-cache scorer",
			info: attrprefix.NewPrefixCacheMatchInfo(4, 10, 16).
				WithCachedBlocksByTier(map[string]int{"gpu": 4}).
				WithTierWeights(map[string]float64{"gpu": 1}),
			scorer: &testPlugin{typedName: fwkplugin.TypedName{Type: "test-scorer", Name: "scorer"}, ScoreRes: 0.5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spanAttrs := pickWithSelectedMatchInfo(t, tt.info, tt.scorer)
			for _, key := range prefixCacheAttributionKeys {
				if _, ok := spanAttrs[key]; ok {
					t.Errorf("%s set, want it omitted", key)
				}
			}
		})
	}
}

func TestRunPickerPluginOmitsTierWeightsWhenAnyTierHasNone(t *testing.T) {
	info := attrprefix.NewPrefixCacheMatchInfo(4, 10, 16).
		WithCachedBlocksByTier(map[string]int{"gpu": 3, "storage": 1}).
		WithTierWeights(map[string]float64{"gpu": 1})

	spanAttrs := pickWithSelectedMatchInfo(t, info, nil)

	if got := spanAttrs[semconv.LLMDEPPPickerPrefixCacheTiersKey].AsStringSlice(); !reflect.DeepEqual(got, []string{"gpu", "storage"}) {
		t.Errorf("prefix_cache.tiers = %v, want [gpu storage]", got)
	}
	// A weights array shorter than the tiers array would misalign the pairs.
	if _, ok := spanAttrs[semconv.LLMDEPPPickerPrefixCacheTierWeightsKey]; ok {
		t.Errorf("%s set although the storage tier has no weight", semconv.LLMDEPPPickerPrefixCacheTierWeightsKey)
	}
}
