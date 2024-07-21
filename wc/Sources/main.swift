import Foundation

let NewLine = "\n".utf8.first!
let Space = " ".utf8.first!

struct Counts {
    var bytes: Int
    var words: Int
    var lines: Int
}

class Counter {
    private var curWordLen = 0
    private var curLineLen = 0
    private var counts = Counts(bytes: 0, words: 0, lines: 0)

    func add(_ ch: UInt8) {
        counts.bytes += 1
        switch ch {
        case NewLine:
            if curWordLen > 0 {
                counts.words += 1
            }
            if curLineLen > 0 {
                counts.lines += 1
            }
            curWordLen = 0
            curLineLen = 0
        case Space:
            if curWordLen > 0 {
                counts.words += 1
            }
            curLineLen += 1
            curWordLen = 0
        default:
            curWordLen += 1
            curLineLen += 1
        }
    }

    func finish() -> Counts {
        // Note that we count trailing words but not trailing lines.
        // This matches the behavior of `wc` on Linux.
        if curWordLen > 0 {
            counts.words += 1
            curWordLen = 0 // ensure idempotency
        }
        return counts
    }
}

func main() {
    let args = CommandLine.arguments;
    if (args.count != 2) {
        print("Usage: \(args[0]) <path-to-file>")
        return
    }
    guard let file = FileHandle(forReadingAtPath: args[1]) else {
        print("failed to open file")
        return
    }
    let bufSize = 1024
    let counter = Counter()
    while true {
        let data: Data
        do {
            if let d = try file.read(upToCount: bufSize) {
                data = d
            } else {
                break
            }
        } catch {
            print("Error reading file: \(error)")
            return
        }
        for x in data {
            counter.add(x)
        }
        if data.count < bufSize {
            break
        }
    }
    print("counts: \(counter.finish())")
}

main()