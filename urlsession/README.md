# urlsession

This is a quick example of using continuations and protocol extensions to create an `async` API out of an API that uses callbacks. In particular, I turn the callback-based `URLSession` API `dataTask()` into an async API. Unfortunately, the `URLSession` doesn't work very well on Linux, but I was still able to test it with an insecure HTTP endpoint.
