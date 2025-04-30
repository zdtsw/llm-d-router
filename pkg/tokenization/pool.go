package tokenization

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/klog/v2"

	"github.com/neuralmagic/llm-d-kv-cache-manager/pkg/prefixstore"

	"k8s.io/client-go/util/workqueue"
)

const defaultWorkers = 5

// Config holds the configuration for the TokenizationPool.
type Config struct {
	WorkersCount int
	*HFTokenizerConfig
}

// DefaultConfig returns a default configuration for the TokenizationPool.
func DefaultConfig() *Config {
	return &Config{
		WorkersCount:      defaultWorkers,
		HFTokenizerConfig: DefaultHFTokenizerConfig(),
	}
}

// Task represents a unit of work for tokenizing a prompt.
type Task struct {
	Prompt    string
	ModelName string
}

// Pool encapsulates the queue, worker pool, and token indexer.
type Pool struct {
	workers int
	queue   workqueue.TypedRateLimitingInterface[Task]
	wg      sync.WaitGroup

	indexer   prefixstore.Indexer
	tokenizer Tokenizer
}

// NewTokenizationPool initializes a TokenizationPool with the specified number
// of workers and the provided Indexer.
func NewTokenizationPool(config *Config, store prefixstore.Indexer) (*Pool, error) {
	if config == nil {
		config = DefaultConfig()
	}

	cachingTokenizer, err := NewCachedHFTokenizer(config.HFTokenizerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	return &Pool{
		workers:   config.WorkersCount,
		queue:     workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[Task]()),
		indexer:   store,
		tokenizer: cachingTokenizer,
	}, nil
}

// AddTask enqueues a new tokenization task.
// This method only enqueues the task and does not start processing it.
func (pool *Pool) AddTask(prompt, modelName string) {
	task := Task{
		Prompt:    prompt,
		ModelName: modelName,
	}
	pool.queue.Add(task)
}

// Run launches worker goroutines that process tasks until the context is
// cancelled.
func (pool *Pool) Run(ctx context.Context) {
	for i := 0; i < pool.workers; i++ {
		pool.wg.Add(1)
		go pool.workerLoop(i)
	}

	<-ctx.Done()

	pool.queue.ShutDown()
	pool.wg.Wait()
}

// workerLoop is the main processing loop for each worker.
func (pool *Pool) workerLoop(_ int) {
	defer pool.wg.Done()
	for {
		task, shutdown := pool.queue.Get()
		if shutdown {
			return
		}

		// Process the task.
		if err := pool.processTask(task); err == nil {
			pool.queue.Forget(task)
		} else {
			pool.queue.AddRateLimited(task)
		}
		pool.queue.Done(task)
	}
}

// processTask tokenizes the prompt, extracts a prefix, and updates the
// indexer.
func (pool *Pool) processTask(task Task) error {
	tokenIds, offsets, err := pool.tokenizer.Encode(task.Prompt, task.ModelName)
	if err != nil {
		klog.Error(err, " failed to encode token", "prompt", task.Prompt, "modelName", task.ModelName)
		return err
	}

	if err := pool.indexer.AddTokenization(task.ModelName, task.Prompt, tokenIds, offsets); err != nil {
		return fmt.Errorf("tokenization failed for model %s: %w", task.ModelName, err)
	}

	return nil
}
