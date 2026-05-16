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

package proxy

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"time"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

var _ = Describe("Mooncake Connector", func() {

	var testInfo *sidecarTestInfo
	var bootstrapServer *httptest.Server

	BeforeEach(func() {
		// Start a mock bootstrap server that returns engine_id
		bootstrapServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"0": {"engine_id": "test-engine-abc123", "worker_addr": {"0": {"0": "10.0.0.1:5000"}}}}`)) //nolint:all
		}))
		DeferCleanup(bootstrapServer.Close)

		// Extract the bootstrap server port
		bootstrapURL, err := url.Parse(bootstrapServer.URL)
		Expect(err).ToNot(HaveOccurred())
		var bootstrapPort int
		_, err = fmt.Sscanf(bootstrapURL.Port(), "%d", &bootstrapPort)
		Expect(err).ToNot(HaveOccurred())

		testInfo = sidecarConnectionTestSetup(KVConnectorMooncake)
		testInfo.proxy.config.MooncakeBootstrapPort = bootstrapPort
	})

	It("should send concurrent requests with correct mooncake kv_transfer_params", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		By("sending a /v1/chat/completions request with prefill header")
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		// Use the bootstrap server's host as the prefill host so engine_id discovery works
		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		// Wait for async prefill request to be fully recorded
		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		// Validate prefill request
		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		prq := prefillReqs[0]

		// Prefill should have kv_transfer_params with do_remote_decode=true
		Expect(prq).To(HaveKey(requestFieldKVTransferParams))
		prefillKVParams, ok := prq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(prefillKVParams[requestFieldDoRemoteDecode]).To(BeTrue())
		Expect(prefillKVParams[requestFieldDoRemotePrefill]).To(BeFalse())
		Expect(prefillKVParams[requestFieldTransferID]).ToNot(BeEmpty())

		// Prefill should have max_tokens=1 and stream=false
		Expect(prq[requestFieldMaxTokens]).To(BeNumerically("==", 1))
		Expect(prq[requestFieldStream]).To(BeFalse())

		// Validate decode request
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		drq := decodeReqs[0]

		// Decode should have kv_transfer_params with do_remote_prefill=true
		Expect(drq).To(HaveKey(requestFieldKVTransferParams))
		decodeKVParams, ok := drq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams[requestFieldDoRemotePrefill]).To(BeTrue())
		Expect(decodeKVParams[requestFieldDoRemoteDecode]).To(BeFalse())
		Expect(decodeKVParams[requestFieldTransferID]).To(HavePrefix("xfer-"))
		Expect(decodeKVParams[requestFieldRemoteEngineID]).To(Equal("test-engine-abc123"))
		Expect(decodeKVParams[requestFieldRemoteBootstrapAddr]).To(ContainSubstring(fmt.Sprintf(":%d", testInfo.proxy.config.MooncakeBootstrapPort)))

		// Transfer IDs must match between prefill and decode
		Expect(decodeKVParams[requestFieldTransferID]).To(Equal(prefillKVParams[requestFieldTransferID]))

		// Decode should preserve original max_tokens
		Expect(drq[requestFieldMaxTokens]).To(BeNumerically("==", 50))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should set max_completion_tokens=1 in prefill and restore in decode", func() {
		By("starting the proxy")
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"max_completion_tokens": 100
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		Eventually(func() int {
			return len(testInfo.prefillHandler.GetCompletionRequests())
		}).Should(Equal(1))

		prefillReqs := testInfo.prefillHandler.GetCompletionRequests()
		Expect(prefillReqs).To(HaveLen(1))
		prq := prefillReqs[0]

		Expect(prq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
		Expect(prq).To(HaveKeyWithValue("max_completion_tokens", BeNumerically("==", 1)))

		decodeReqs := testInfo.decodeHandler.GetCompletionRequests()
		Expect(decodeReqs).To(HaveLen(1))
		drq := decodeReqs[0]

		Expect(drq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 50)))
		Expect(drq).To(HaveKeyWithValue("max_completion_tokens", BeNumerically("==", 100)))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})

	It("should not panic when prefill response is slower than decode response", func() {
		// Stop previously injected servers
		testInfo.decodeBackend.Close()
		testInfo.prefillBackend.Close()

		var prefillFinished atomic.Bool

		slowPrefill := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.prefillHandler.ServeHTTP(w, r)
			time.Sleep(300 * time.Millisecond)
			prefillFinished.Store(true)
		})
		testInfo.prefillBackend = httptest.NewServer(slowPrefill)

		fastDecode := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			testInfo.decodeHandler.ServeHTTP(w, r)
		})
		testInfo.decodeBackend = httptest.NewServer(fastDecode)
		testInfo.decodeURL, _ = url.Parse(testInfo.decodeBackend.URL)

		var bootstrapPort int
		bURL, _ := url.Parse(bootstrapServer.URL)
		_, _ = fmt.Sscanf(bURL.Port(), "%d", &bootstrapPort)

		cfg := Config{
			Port:                  "0",
			DecoderURL:            testInfo.decodeURL,
			KVConnector:           KVConnectorMooncake,
			MooncakeBootstrapPort: bootstrapPort,
		}
		testInfo.proxy = NewProxy(cfg)

		go func() {
			defer GinkgoRecover()
			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())
			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		proxyBaseAddr := "http://" + testInfo.proxy.addr.String()

		body := `{"model": "Qwen", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}`
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())

		prefillHostPort := testInfo.prefillBackend.URL[len("http://"):]
		req.Header.Add(routing.PrefillEndpointHeader, prefillHostPort)

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(200))

		time.Sleep(500 * time.Millisecond)

		Expect(prefillFinished.Load()).To(BeTrue())
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))

		testInfo.cancelFn()
		<-testInfo.stoppedCh
	})
})
