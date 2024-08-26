public final class Tensor {

    public final class BackwardHandle {
        private var addGrad: ((Tensor?) -> Void)?

        init() {
            self.addGrad = {_ in ()}
        }

        init(addGrad: @escaping (Tensor?) -> Void) {
            self.addGrad = addGrad
        }

        public func backward(grad: Tensor) {
            assert(!grad.needsGrad, "second-order gradients are not supported")
            assert(self.addGrad != nil, "cannot re-use backward handle")
            self.addGrad!(grad)
            self.addGrad = nil
        }

        deinit {
            if let addGrad = addGrad {
                addGrad(nil)
            }
        }
    }

    public let data: [Float]
    public let shape: [Int]
    public let needsGrad: Bool

    private var backwardImpl: ((Tensor) -> Void)?

    // State used during backward pass to accumulate gradients.
    private var curGrad: Tensor? = nil
    private var numBackwardHandles: Int = 0

    public init(data: [Float], shape: [Int], backwardImpl: ((Tensor) -> Void)? = nil) {
        assert(
            data.count == shape.product(),
            "data of size \(data.count) incompatible with shape \(shape)"
        )
        self.data = data
        self.shape = shape
        self.backwardImpl = backwardImpl
        self.needsGrad = backwardImpl != nil
    }

    public convenience init(zerosLike: Tensor) {
        self.init(constant: 0, like: zerosLike)
    }

    public convenience init(onesLike: Tensor) {
        self.init(constant: 1, like: onesLike)
    }

    public convenience init(zeros shape: [Int]) {
        self.init(constant: 0, shape: shape)
    }

    public convenience init(ones shape: [Int]) {
        self.init(constant: 1, shape: shape)
    }

    public convenience init(constant: Float, like: Tensor) {
        self.init(constant: constant, shape: like.shape)
    }

    public init(constant: Float, shape: [Int]) {
        self.data = Array(repeating: constant, count: shape.product())
        self.shape = shape
        self.needsGrad = false
    }

    public func item() -> Float {
        assert(data.count == 1, "cannot call item() on Tensor of shape \(shape)")
        return data[0]
    }

    public func noGrad() -> Tensor {
        return Tensor(data: data, shape: shape)
    }

    public func onGrad(_ action: @escaping (Tensor) -> Void) -> Tensor {
        if !needsGrad {
            return Tensor(data: data, shape: shape, backwardImpl: action)
        }
        let handle = self.saveForBackward()
        return Tensor(data: data, shape: shape) { grad in 
            action(grad)
            handle.backward(grad: grad)
        }
    }

    public func reshape(_ newShape: [Int]) -> Tensor {
        if shape == newShape {
            return self
        }
        assert(shape.product() == newShape.product())
        if !needsGrad {
            return Tensor(data: data, shape: newShape)
        } else {
            let handle = self.saveForBackward()
            return Tensor(data: data, shape: newShape) { grad in
                handle.backward(grad: grad.reshape(self.shape))
            }
        }
    }

    public static func * (lhs: Tensor, rhs: Float) -> Tensor {
        let newData = Array(lhs.data.map({x in x * rhs}))
        if !lhs.needsGrad {
            return Tensor(data: newData, shape: lhs.shape)
        } else {
            let lhsHandle = lhs.saveForBackward()
            return Tensor(data: newData, shape: lhs.shape) { grad in
                lhsHandle.backward(grad: grad * rhs)
            }
        }
    }

    public static func + (lhs: Tensor, rhs: Float) -> Tensor {
        let newData = Array(lhs.data.map({x in x + rhs}))
        if !lhs.needsGrad {
            return Tensor(data: newData, shape: lhs.shape)
        } else {
            let lhsHandle = lhs.saveForBackward()
            return Tensor(data: newData, shape: lhs.shape, backwardImpl: lhsHandle.backward)
        }
    }

    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        assert(
            lhs.shape == rhs.shape,
            "shape mismatch for + operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
        )
        let newData = Array(zip(lhs.data, rhs.data).map({x in x.0 + x.1}))
        if !lhs.needsGrad && !rhs.needsGrad {
            return Tensor(data: newData, shape: lhs.shape)
        } else {
            let lhsHandle = lhs.saveForBackward()
            let rhsHandle = rhs.saveForBackward()
            return Tensor(data: newData, shape: lhs.shape) { grad in 
                lhsHandle.backward(grad: grad)
                rhsHandle.backward(grad: grad)
            }
        }
    }

    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        assert(
            lhs.shape == rhs.shape,
            "shape mismatch for * operator: lhs=\(lhs.shape) rhs=\(rhs.shape)"
        )
        let newData = Array(zip(lhs.data, rhs.data).map({x in x.0 * x.1}))
        if !lhs.needsGrad && !rhs.needsGrad {
            return Tensor(data: newData, shape: lhs.shape)
        } else {
            let lhsHandle = lhs.saveForBackward()
            let rhsHandle = rhs.saveForBackward()
            return Tensor(data: newData, shape: lhs.shape) { grad in 
                lhsHandle.backward(grad: grad * rhs.noGrad())
                rhsHandle.backward(grad: grad * lhs.noGrad())
            }
        }
    }

    public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return lhs + -1*rhs
    }

    public static func - (lhs: Tensor, rhs: Float) -> Tensor {
        return lhs + -rhs
    }

    prefix public static func - (t: Tensor) -> Tensor {
        return t * -1
    }

    public static func - (lhs: Float, rhs: Tensor) -> Tensor {
        return lhs + -rhs
    }

    public static func + (lhs: Float, rhs: Tensor) -> Tensor {
        return rhs + lhs
    }

    public static func * (lhs: Float, rhs: Tensor) -> Tensor {
        return rhs * lhs
    }

    public static func / (lhs: Tensor, rhs: Tensor) -> Tensor {
        return lhs * rhs.pow(-1)
    }

    public static func / (lhs: Float, rhs: Tensor) -> Tensor {
        return lhs * rhs.pow(-1)
    }

    public static func / (lhs: Tensor, rhs: Float) -> Tensor {
        return lhs * (1 / rhs)
    }

    public static func == (lhs: Tensor, rhs: Float) -> Tensor {
        return Tensor(data: Array(lhs.data.map { x in x == rhs ? 1.0 : 0.0 }), shape: lhs.shape)
    }

    public static func == (lhs: Float, rhs: Tensor) -> Tensor {
        return rhs == lhs
    }

    public static func == (lhs: Tensor, rhs: Tensor) -> Tensor {
        assert(lhs.shape == rhs.shape, "incompatible operands for ==: \(lhs.shape) != \(rhs.shape)")
        return Tensor(
            data: Array(zip(lhs.data, rhs.data).map { (x, y) in x == y ? 1.0 : 0.0}),
            shape: lhs.shape
        )
    }

    public func backward(grad: Tensor? = nil) {
        let grad = if let grad = grad {
            grad
        } else {
            Tensor(onesLike: self)
        }

        assert(numBackwardHandles == 0, "cannot call backward() on tensor that is used elsewhere")
        self.saveForBackward().backward(grad: grad)
    }

    public func saveForBackward() -> BackwardHandle {
        if !self.needsGrad {
            return BackwardHandle()
        }
        assert(self.backwardImpl != nil, "cannot backward a second time")
        numBackwardHandles += 1
        return BackwardHandle(addGrad: { [self] grad in
            assert(numBackwardHandles > 0)
            if let grad = grad {
                assert(
                    grad.shape == shape,
                    "gradient shape \(grad.shape) must match tensor shape \(shape)"
                )
                if let cg = curGrad {
                    curGrad = cg + grad
                } else {
                    curGrad = grad
                }
            }
            numBackwardHandles -= 1
            if numBackwardHandles == 0 && curGrad != nil {
                let bwd = backwardImpl!
                backwardImpl = nil
                if let grad = curGrad {
                    bwd(grad)
                    curGrad = nil
                } else {
                    bwd(Tensor(zerosLike: self))
                }
            }
        })
    }

}
