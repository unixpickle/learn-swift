import Foundation
import FoundationNetworking

// HTTPS doesn't seem to work at all on Linux.
let WebsiteURL = "http://www.aqnichol.com/"

protocol AsyncData {
    func dataAsync(with: URL) async throws -> (Data, URLResponse);
}

struct InconsistentResultError: Error {
}

extension URLSession: AsyncData {
    func dataAsync(with: URL) async throws -> (Data, URLResponse) {
        // Hack to introduce a short timeout
        let req = URLRequest(url: with, cachePolicy: .useProtocolCachePolicy, timeoutInterval: 5.0)

        return try await withCheckedThrowingContinuation({continuation in
            let req = self.dataTask(with: req) {data, resp, err in
                if let err = err {
                    continuation.resume(throwing: err)
                } else {
                    if let data = data, let resp = resp {
                        continuation.resume(returning: (data, resp))
                    } else {
                        continuation.resume(throwing: InconsistentResultError())
                    }
                }
            }
            req.resume()
        })
    }
}

@main
struct Main {
    static func main() async {
        do {
            let (data, response) = try await URLSession.shared.dataAsync(with: URL(string: WebsiteURL)!)
            if let x = response as? HTTPURLResponse {
                print("status code: \(x.statusCode)")
            }
            let resString = String(decoding: data, as: UTF8.self);
            let lines = resString.split(separator: "\n");
            print("first line of result: \(lines[0])")
        } catch {
            print("error: \(error)")
        }
    }
}
