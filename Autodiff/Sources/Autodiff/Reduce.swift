extension Tensor {
    func sum(axis: Int? = nil) -> Tensor {
        if let axis = axis {
            let trueAxis = (axis < 0 ? axis + shape.count : axis)
            assert(
                trueAxis >= 0 && trueAxis < shape.count,
                "axis \(axis) out of bounds for tensor shape \(shape)"
            )
            let outerCount = shape[..<trueAxis].product()
            let reduceCount = shape[trueAxis]
            let innerCount = shape[(trueAxis+1)...].product()
            var newData = Array(repeating: Float(0), count: outerCount * innerCount)
            for i in 0..<outerCount {
                let batchOffset = i*reduceCount*innerCount
                for j in 0..<innerCount {
                    var sum = Float(0)
                    for k in 0..<reduceCount {
                        sum += data[batchOffset + k*innerCount + j]
                    }
                    newData[i*innerCount + j] = sum
                }
            }
            let newShape = Array(shape[..<trueAxis] + shape[(trueAxis+1)...])
            if !needsGrad {
                return Tensor(data: newData, shape: newShape)
            } else {
                let handle = self.saveForBackward()
                return Tensor(data: newData, shape: newShape) { grad in
                    var repeatedGrad = Array(repeating: Float(0), count: self.data.count)
                    for i in 0..<outerCount {
                        let batchOffset = i*reduceCount*innerCount
                        for j in 0..<innerCount {
                            let src = grad.data[i*innerCount + j]
                            for k in 0..<reduceCount {
                                repeatedGrad[batchOffset + k*innerCount + j] = src
                            }
                        }
                    }
                    handle.backward(grad: Tensor(data: repeatedGrad, shape: self.shape))
                }
            }
        } else {
            let newData = [self.data.sum()]
            if !needsGrad {
                return Tensor(data: newData, shape: [])
            } else {
                let handle = self.saveForBackward()
                return Tensor(data: newData, shape: []) { grad in
                    assert(grad.data.count == 1)
                    handle.backward(
                        grad: Tensor(
                            data: Array(repeating: grad.data[0], count: self.data.count),
                            shape: self.shape
                        )
                    )
                }
            }
        }
    }

    func min(axis: Int? = nil) -> Tensor {
        return maxOrMin(isMax: false, axis: axis)
    }

    func max(axis: Int? = nil) -> Tensor {
        return maxOrMin(isMax: true, axis: axis)
    }

    private func maxOrMin(isMax: Bool, axis: Int? = nil) -> Tensor {
        if let axis = axis {
            let trueAxis = (axis < 0 ? axis + shape.count : axis)
            assert(
                trueAxis >= 0 && trueAxis < shape.count,
                "axis \(axis) out of bounds for tensor shape \(shape)"
            )
            assert(shape[trueAxis] > 0, "cannot perform this reduction along empty axis")
            let outerCount = shape[..<trueAxis].product()
            let reduceCount = shape[trueAxis]
            let innerCount = shape[(trueAxis+1)...].product()
            var newData = Array(repeating: Float(0), count: outerCount * innerCount)
            var indices = Array(repeating: 0, count: outerCount * innerCount)
            for i in 0..<outerCount {
                let batchOffset = i*reduceCount*innerCount
                for j in 0..<innerCount {
                    var reducedValue = Float(0)
                    var chosenIndex = 0
                    for k in 0..<reduceCount {
                        let v = data[batchOffset + k*innerCount + j]
                        if k == 0 || ((isMax && v > reducedValue) || (!isMax && v < reducedValue)) {
                            reducedValue = v
                            chosenIndex = k
                        }
                    }
                    newData[i*innerCount + j] = reducedValue
                    indices[i*innerCount + j] = batchOffset + chosenIndex*innerCount + j
                }
            }
            let newShape = Array(shape[..<trueAxis] + shape[(trueAxis+1)...])
            if !needsGrad {
                return Tensor(data: newData, shape: newShape)
            } else {
                let handle = self.saveForBackward()
                return Tensor(data: newData, shape: newShape) { grad in
                    handle.backward(grad: grad.scatter(outShape: self.shape, dstIndices: indices))
                }
            }
        } else {
            return self.reshape([shape.product()]).maxOrMin(isMax: isMax, axis: 0)
        }
    }   
}