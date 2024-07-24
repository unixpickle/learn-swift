import SwiftCrossUI
import GtkBackend

class Recipe {
    var ingredients: [String] = []
    var steps: [String] = []
}

class RecipeState: Observable {
    @Observed var url: String = ""
    @Observed var loading: Bool = false
    @Observed var errorMessage: String?
    @Observed var recipe: Recipe?
}

@main
struct RecipeApp: App {
    let identifier = "com.aqnichol.RecipeApp"

    let state = RecipeState()

    var body: some Scene {
        WindowGroup("RecipeApp") {
            VStack{
                HStack {
                    TextField("URL to page", state.$url)
                    Button("Lookup") { lookup() }
                }.padding(10)
                if state.loading {
                    Text("Loading...").padding([.leading, .trailing, .bottom], 10)
                } else {
                    Text("Content here...").padding([.leading, .trailing, .bottom], 10)
                }
            }
        }
    }

    func lookup() {
        Task.detached {
            await self.runOnMain { state.loading = true }
            try await Task.sleep(nanoseconds:1000000000)
            await self.runOnMain { state.loading = false }
        }
    }

    func runOnMain<T>(fn: @escaping () -> T) async -> T {
        return await withUnsafeContinuation() { continuation in 
            // Incredibly massive hack to run in the main UI thread.
            let x = Publisher()

            // We must maintain a reference to this because deinit()
            // removes the observer.
            var c: Cancellable?

            c = x.observe {
                c!.cancel()
                continuation.resume(returning: fn())
            }
            x.send()
        }
    }
}



