import Foundation
import SwiftCrossUI
import GtkBackend
import AsyncHTTPClient

class Recipe {
    var ingredients: [String]
    var steps: [String]

    init(ingredients: [String], steps: [String]) {
        self.ingredients = ingredients
        self.steps = steps
    }
}

enum RecipeError: Error {
    case decodeBody
    case findJsonBlob
    case decodeJson
    case extractRecipe
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
                    Button("Lookup") { Task.detached { await lookup() } }
                }.padding(10)
                if state.loading {
                    Text("Loading...").padding([.leading, .trailing, .bottom], 10)
                } else {
                    if let msg = state.errorMessage {
                        Text(msg).padding([.leading, .trailing, .bottom], 10)
                    } else if let contents = state.recipe {
                        HStack {
                            Text("Ingredients:\n\n\(String(contents.ingredients.joined(by: "\n")))").frame(minWidth: 200)
                            Text("Steps:\n\n\(String(contents.steps.joined(by: "\n")))").frame(minWidth: 400)
                        }.padding([.leading, .trailing, .bottom], 10)
                    } else {
                        Text("Enter a URL to lookup recipe.").padding([.leading, .trailing, .bottom], 10)
                    }
                }
            }
        }
    }

    func lookup() async {
        let url = await self.runOnMain {
            state.loading = true
            return state.url
        }
        do {
            let recipe = try await self.fetchRecipe(url: url)
            await self.runOnMain { state.recipe = recipe }
        } catch {
            await self.runOnMain {
                state.errorMessage = "Failed to fetch: \(error)"
            }
        }
        await self.runOnMain { state.loading = false }
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

    func fetchRecipe(url: String) async throws -> Recipe {
        let request = HTTPClientRequest(url: url)
        let response = try await HTTPClient.shared.execute(request, timeout: .seconds(30))
        let body = try await response.body.collect(upTo: 1 << 24)
        // TODO: parse it here
        guard let bodyStr = body.getString(at: 0, length: body.readableBytes) else {
            throw RecipeError.decodeBody
        }
        guard let startIdx = bodyStr.range(of: "<script type=\"application/ld+json\">")?.upperBound else {
            throw RecipeError.findJsonBlob
        }
        let fromStart = bodyStr[startIdx...]
        guard let endIdx = fromStart.range(of: "</script>")?.lowerBound else {
            throw RecipeError.findJsonBlob
        }
        guard let jsonBlob = fromStart[..<endIdx].data(using: .utf8) else {
            throw RecipeError.findJsonBlob
        }
        do {
            let parsed = try JSONDecoder().decode(RecipeObject.self, from: jsonBlob)
            return Recipe(
                ingredients: parsed.recipeIngredient,
                steps: parsed.recipeInstructions.map() { $0.text }
            )
        } catch {
            throw RecipeError.decodeJson
        }
    }
}

struct RecipeStep: Decodable {
    var text: String
}

struct RecipeObject: Decodable {
    var name: String
    var description: String
    var recipeIngredient: [String]
    var recipeInstructions: [RecipeStep]
}
