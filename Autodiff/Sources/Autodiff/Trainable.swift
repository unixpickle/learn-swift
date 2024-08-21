class Trainable {

    @propertyWrapper
    final class Parameter {
        static subscript<T: Trainable>(
            _enclosingInstance instance: T,
            wrapped wrappedKeyPath: ReferenceWritableKeyPath<T, Tensor>,
            storage storageKeyPath: ReferenceWritableKeyPath<T, Parameter>
        ) -> Tensor {
            get {
                let param = instance[keyPath: storageKeyPath]
                let rawTensor = instance[keyPath: storageKeyPath].data!
                return rawTensor.onGrad { g in
                    if let existingGrad = param.grad {
                        param.grad = existingGrad + g
                    } else {
                        param.grad = g
                    }
                }
            }
            set {
                let param = instance[keyPath: storageKeyPath]
                param.data = newValue
                instance.registeredParams[param.name] = param
            }
        }

        @available(*, unavailable,
            message: "@Parameter can only be applied to classes"
        )
        var wrappedValue: Tensor {
            get { fatalError() }
            set { fatalError() }
        }

        let name: String
        var data: Tensor?
        var grad: Tensor?

        var projectedValue: Parameter { self }

        init(name: String) {
            self.name = name
        }
    }

    @propertyWrapper
    final class Child<Value: Trainable> {
        static subscript<T: Trainable>(
            _enclosingInstance instance: T,
            wrapped wrappedKeyPath: ReferenceWritableKeyPath<T, Value>,
            storage storageKeyPath: ReferenceWritableKeyPath<T, Child>
        ) -> Value {
            get {
                instance[keyPath: storageKeyPath].value!
            }
            set {
                let child = instance[keyPath: storageKeyPath]
                child.value = newValue
                instance.registeredChildren[child.name] = newValue
            }
        }

        @available(*, unavailable,
            message: "@Child can only be applied to classes"
        )
        var wrappedValue: Value {
            get { fatalError() }
            set { fatalError() }
        }

        let name: String
        private var value: Value?

        init(name: String) {
            self.name = name
        }
    }

    internal var registeredParams = [String: Parameter]()
    internal var registeredChildren = [String: Trainable]()

    public var parameters: [(String, Parameter)] {
        var results = Array(registeredParams)
        for (name, child) in registeredChildren {
            results += child.parameters.map { (subName, param) in
                ("\(name).\(subName)", param)
            }
        }
        return results.sorted(by: { $0.0 < $1.0 })
    }

}

class TrainableArray<T: Trainable>: Trainable {
    let children: [Trainable]

    init(_ children: [T]) {
        self.children = children
        super.init()
        for (i, ch) in children.enumerated() {
            self.registeredChildren[String(i)] = ch
        }
    }
}
