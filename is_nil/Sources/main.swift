// This version does not work
//     error: generic parameter 'Wrapped' could not be inferred in cast to 'Optional'
//
// func isNil<Value>(_ x: Value) -> Bool {
//     return if let o = x as? Optional, o == nil {
//         true
//     } else {
//         false
//     }
// }

// Example taken from the blog post.

protocol MyOptional {
    var isNil: Bool { get }
}

extension Optional: MyOptional {
    var isNil: Bool { self == nil }
}

func isNil<Value>(_ x: Value) -> Bool {
    return if let o = x as? MyOptional, o.isNil {
        true
    } else {
        false
    }
}

func isNil2<Value>(_ x: Value) -> Bool {
    return if let x = x as? Optional<Int> {
        x == nil
    } else {
        false
    }
}

func main() {
    let w: Int? = nil;
    let x: String? = nil;
    let y: Int = 3;
    let z: String? = "hi";
    print("should be nil: \(isNil(w)), \(isNil2(w))")
    print("should be nil: \(isNil(x)), \(isNil2(x))")
    print("should not be nil: \(isNil(y)), \(isNil2(y))")
    print("should not be nil: \(isNil(z)), \(isNil2(z))")

    if x as? Optional<Int> != nil {
        print("a nil String? can be converted to Optional<Int>")
    }
}

main()
