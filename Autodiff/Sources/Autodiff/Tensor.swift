final class Tensor {
    let data: [Float]
    let shape: [Int]
    let needsGrad: Bool

    private var backwardImpl: ((Tensor) -> Void)?

    // State used during backward pass to accumulate gradients.
    private var curGrad: Tensor? = nil
    private var numBackwardHandles: Int = 0

    init(data: [Float], shape: [Int], backwardImpl: ((Tensor) -> Void)? = nil) {
        assert(
            data.count == shape.product(),
            "data of size \(data.count) incompatible with shape \(shape)"
        )
        self.data = data
        self.shape = shape
        self.backwardImpl = backwardImpl
        self.needsGrad = backwardImpl != nil
    }

    convenience init(zerosLike: Tensor) {
        self.init(constant: 0, like: zerosLike)
    }

    convenience init(onesLike: Tensor) {
        self.init(constant: 1, like: onesLike)
    }

    convenience init(zeros shape: [Int]) {
        self.init(constant: 0, shape: shape)
    }

    convenience init(ones shape: [Int]) {
        self.init(constant: 1, shape: shape)
    }

    convenience init(constant: Float, like: Tensor) {
        self.init(constant: constant, shape: like.shape)
    }

    init(constant: Float, shape: [Int]) {
        self.data = Array(repeating: constant, count: shape.product())
        self.shape = shape
        self.needsGrad = false
    }

    func noGrad() -> Tensor {
        return Tensor(data: data, shape: shape)
    }

    func onGrad(action: @escaping (Tensor) -> Void) -> Tensor {
        if !needsGrad {
            return Tensor(data: data, shape: shape, backwardImpl: action)
        }
        let handle = self.saveForBackward()
        return Tensor(data: data, shape: shape) { grad in 
            action(grad)
            handle.backward(grad: grad)
        }
    }

    static func * (lhs: Tensor, rhs: Float) -> Tensor {
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

    static func + (lhs: Tensor, rhs: Float) -> Tensor {
        let newData = Array(lhs.data.map({x in x + rhs}))
        if !lhs.needsGrad {
            return Tensor(data: newData, shape: lhs.shape)
        } else {
            let lhsHandle = lhs.saveForBackward()
            return Tensor(data: newData, shape: lhs.shape, backwardImpl: lhsHandle.backward)
        }
    }

    static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
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

    static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
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

    static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
        return lhs + -1*rhs
    }

    static func - (lhs: Tensor, rhs: Float) -> Tensor {
        return lhs + -rhs
    }

    prefix static func - (t: Tensor) -> Tensor {
        return t * -1
    }

    static func - (lhs: Float, rhs: Tensor) -> Tensor {
        return lhs + -rhs
    }

    static func + (lhs: Float, rhs: Tensor) -> Tensor {
        return rhs + lhs
    }

    static func * (lhs: Float, rhs: Tensor) -> Tensor {
        return rhs * lhs
    }

    func backward(grad: Tensor) {
        assert(numBackwardHandles == 0, "cannot call backward() on tensor that is used elsewhere")
        self.saveForBackward().backward(grad: grad)
    }

    func saveForBackward() -> TensorBackwardHandle {
        if !self.needsGrad {
            return TensorBackwardHandle()
        }
        assert(self.backwardImpl != nil, "cannot backward a second time")
        numBackwardHandles += 1
        return TensorBackwardHandle(addGrad: { [self] grad in
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
            if numBackwardHandles == 0 {
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

final class TensorBackwardHandle {
    private var addGrad: ((Tensor?) -> Void)?

    init() {
        self.addGrad = {_ in ()}
    }

    init(addGrad: @escaping (Tensor?) -> Void) {
        self.addGrad = addGrad
    }

    func backward(grad: Tensor) {
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
