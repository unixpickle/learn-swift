import Foundation

extension Tensor {
    func elemwise(_ f: (Float) -> Float, fgrad: ((Float) -> Float)? = nil) -> Tensor {
        let newData = Array(data.map(f))
        if !needsGrad {
            return Tensor(data: newData, shape: shape)
        } else {
            assert(fgrad != nil, "operation has no gradient implementation")
            let handle = self.saveForBackward()
            return Tensor(data: newData, shape: shape) { grad in
                handle.backward(grad: self.noGrad().elemwise(fgrad!) * grad)
            }
        }
    }

    func sin() -> Tensor {
        elemwise(Foundation.sin, fgrad: Foundation.cos)
    }

    func cos() -> Tensor {
        elemwise(Foundation.cos, fgrad: {-Foundation.sin($0)})
    }

    func pow(_ exponent: Float) -> Tensor {
        elemwise {x in
            Foundation.pow(x, exponent)
        } fgrad: { x in 
            exponent * Foundation.pow(x, exponent - 1)
        }
    }

    func sqrt() -> Tensor {
        pow(0.5)
    }

    func rsqrt() -> Tensor {
        pow(-0.5)
    }

    func exp() -> Tensor {
        elemwise(Foundation.exp, fgrad: Foundation.exp)
    }

    func log() -> Tensor {
        elemwise(Foundation.log, fgrad: { 1 / $0 })
    }

    func sigmoid() -> Tensor {
        return elemwise(safeSigmoid) { x in
            let s = safeSigmoid(x)
            return s * (1-s)
        }
    }

    func relu() -> Tensor {
        return elemwise { $0 < 0 ? 0 : $0 } fgrad: { $0 < 0 ? 0 : 1}
    }

    func tanh() -> Tensor {
        2 * (2*self).sigmoid() - 1
    }

    func gelu() -> Tensor {
        0.5 * self * (1 + (0.797884561 * (self + 0.044715 * self.pow(3))).tanh())
    }

    func silu() -> Tensor {
        return self * self.sigmoid()
    }
}

private func safeSigmoid(_ x: Float) -> Float {
    if x < -20 {
        return 0
    } else if x > 20 {
        return 1
    } else {
        return 1 / (1 + exp(-x))
    }
}