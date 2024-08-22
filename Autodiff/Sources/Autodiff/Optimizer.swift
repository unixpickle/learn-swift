import Foundation

class Optimizer {
    typealias Parameter = Trainable.Parameter

    let parameters: [String: Parameter]

    init(_ parameters: [(String, Parameter)]) {
        self.parameters = [String: Parameter](uniqueKeysWithValues: parameters)
    }

    func clearGrads() {
        for p in self.parameters.values {
            p.grad = nil
        }
    }

    func step() {
        preconditionFailure("Method not implemented")
    }
}

class Adam: Optimizer {
    public var lr: Float
    public var beta1: Float
    public var beta2: Float
    public var eps: Float

    public var stepIndex: [String: Int] = [:]
    public var moment1: [String: Tensor] = [:]
    public var moment2: [String: Tensor] = [:]

    init(_ parameters: [(String, Parameter)], lr: Float, beta1: Float = 0.9, beta2: Float = 0.999, eps: Float = 1e-8) {
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        super.init(parameters)
    }

    override func step() {
        for (name, param) in parameters {
            guard let grad = param.grad else {
                continue
            }
            let t = stepIndex[name] ?? 1
            stepIndex[name] = t + 1

            var mt = moment1[name] ?? Tensor(zerosLike: grad)
            var vt = moment2[name] ?? Tensor(zerosLike: grad)
            mt = beta1 * mt + (1 - beta1) * grad
            vt = beta2 * vt + (1 - beta2) * grad.pow(2)
            moment1[name] = mt
            moment2[name] = vt
            mt = mt / (1 - pow(beta1, Float(t)))
            vt = vt / (1 - pow(beta2, Float(t)))
            param.data! = param.data! - lr * mt / (vt.sqrt() + eps)
        }
    }
}