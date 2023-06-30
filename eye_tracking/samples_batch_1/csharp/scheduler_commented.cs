/*************************************************************************************************/
#nullable enable
using System;
using System.Threading.Tasks;

class ScheduledTask<T> where T : struct {
    private readonly int expirationMs;
    private readonly Func<Task<T?>> producer;
    private TaskCompletionSource<T?>? promise = null;
    private Nullable<T> result = null;

    public ScheduledTask(Func<Task<T?>> producer, int expirationMs = int.MaxValue) {
        this.expirationMs = expirationMs;
        this.producer = producer;
    }

    public async Task<T?> Run() {
        if (this.promise == null) {
            this.promise = new TaskCompletionSource<T?>();
            this.promise.SetResult(await this.producer());
            // Not awaiting so it won't block.
            this.StoreResult(this.promise.Task)
                // clear after expiration date
                .ContinueWith(t => {
                    if (this.expirationMs < int.MaxValue && this.promise != null) {
                        Task.Delay(this.expirationMs).ContinueWith(t2 => {
							this.promise = null;
							this.result = default(T?);
						});
                    }
                });
        }
        return await this.promise.Task;
    }

    private async Task StoreResult(Task<T?> promise) {
        try {
            this.result = await promise;
        } finally {
            if (this.result == null) {
                this.promise = null; // allow retry
            }
        }
    }

    public T? Value() {
        return this.result;
    }
}