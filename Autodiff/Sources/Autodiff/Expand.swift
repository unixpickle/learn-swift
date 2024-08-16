extension Tensor {

    func expand(shape newShape: [Int]) -> Tensor {
        assert(
            newShape.count >= shape.count,
            "cannot broadcast shape \(shape) to shorter shape \(newShape)"
        )
        var result = self
        for i in 0..<shape.count {
            let axis = shape.count - (i + 1)
            let oldValue = shape[axis]
            let newValue = newShape[newShape.count - (i + 1)]
            if newValue != oldValue {
                assert(
                    oldValue == 1,
                    "axis \(axis) cannot expand from size \(oldValue) to \(newValue)"
                )
                result = result.repeating(axis: axis, count: newValue)
            }
        }
        if newShape.count > shape.count {
            let repeats = newShape[..<(newShape.count - shape.count)].product()
            result = result.repeating(axis: 0, count: repeats)
        }
        return result.reshape(newShape)
    }

    func repeating(axis: Int, count: Int) -> Tensor {
        let trueAxis = (axis < 0 ? axis + shape.count : axis)
        assert(
            trueAxis >= 0 && trueAxis < shape.count,
            "axis \(axis) out of bounds for tensor shape \(shape)"
        )
        let outerCount = shape[..<trueAxis].product()
        let innerCount = shape[trueAxis...].product()
        var newData = Array<Float>()
        for i in 0..<outerCount {
            let chunk = data[i*innerCount..<((i+1)*innerCount)]
            for _ in 0..<count {
                newData += chunk
            }
        }
        let newShape = Array(shape[..<trueAxis] + [count * shape[trueAxis]] + shape[(trueAxis+1)...])
        if !needsGrad {
            return Tensor(data: newData, shape: newShape)
        } else {
            let handle = self.saveForBackward()
            return Tensor(data: newData, shape: newShape) { grad in
                handle.backward(grad: grad.sum(axis: trueAxis, keepdims: true))
            }
        }
    }

}