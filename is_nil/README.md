# is_nil

This is a quick experiment that I wanted to investigate after reading [this blog post](https://www.swiftbysundell.com/articles/property-wrappers-in-swift/).

The author does the following

```swift
private protocol AnyOptional {
    var isNil: Bool { get }
}

extension Optional: AnyOptional {
    var isNil: Bool { self == nil }
}

@propertyWrapper struct UserDefaultsBacked<Value> {
    var wrappedValue: Value {
        get { ... }
        set {
            if let optional = newValue as? AnyOptional, optional.isNil {
                storage.removeObject(forKey: key)
            } else {
                storage.setValue(newValue, forKey: key)
            }
        }
    }
    
    ...
}
```

I don't understand why AnyProtocol is necessary.

# Result

AnyProtocol is necessary because you can use `as? Protocol` with it. You cannot do `as? Optional` because `Optional` apparently has a generic parameter which cannot be inferred (I assume the contained value).

However, you **can** do `as? Optional<Int>`, and *any* `nil` value can seemingly be cast to this type. E.g. this works for both `String?` and `Int?`, in a seemingly incredibly hacky way!

```swift
func isNil2<Value>(_ x: Value) -> Bool {
    return if let x = x as? Optional<Int> {
        // Actually works for all contained types.
        x == nil
    } else {
        false
    }
}
```
