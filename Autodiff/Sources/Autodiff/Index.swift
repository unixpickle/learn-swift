protocol TensorIndex {
    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int])
}

struct AllIndices: TensorIndex {
    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        let numel = shapeProduct(inShape)
        return (Array(0..<numel), inShape)
    }
}

extension Int: TensorIndex {
    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let idx = self < 0 ? inShape[0] + self : self
        let (innerIndices, _) = AllIndices().tensorSliceIndices(forShape: Array(inShape[1...]))
        let offset = idx*innerIndices.count
        return (Array(innerIndices.map({$0 + offset})), Array(inShape[1...]))
    }
}

extension Range<Int>: TensorIndex {
    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let start = self.lowerBound < 0 ? inShape[0] + self.lowerBound : self.lowerBound
        let end = self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound
        var result: [Int] = Array()
        for i in start..<end {
            result += i.tensorSliceIndices(forShape: inShape).0
        }
        return (result, [end - start] + inShape[1...])
    }
}

extension Tensor {
    func scatter(outShape: [Int], dstIndices: [Int]) -> Tensor {
        assert(dstIndices.count == data.count)
        var newData: [Float] = Array(repeating: 0, count: shapeProduct(outShape))
        for (src, dstIdx) in zip(data, dstIndices) {
            newData[dstIdx] = src
        }
        if !needsGrad {
            return Tensor(data: newData, shape: outShape)
        } else {
            let handle = self.saveForBackward()
            return Tensor(data: newData, shape: outShape) { grad in
                handle.backward(grad: grad.gather(outShape: self.shape, srcIndices: dstIndices))
            }
        }
    }

    func gather(outShape: [Int], srcIndices: [Int]) -> Tensor {
        var newData: [Float] = Array(repeating: 0, count: shapeProduct(outShape))
        assert(srcIndices.count == newData.count)
        for (i, srcIdx) in srcIndices.enumerated() {
            newData[i] = data[srcIdx]
        }
        if !needsGrad {
            return Tensor(data: newData, shape: outShape)
        } else {
            let handle = self.saveForBackward()
            return Tensor(data: data, shape: outShape) { grad in
                handle.backward(grad: grad.scatter(outShape: self.shape, dstIndices: srcIndices))
            }
        }
    }

    subscript(index: TensorIndex) -> Tensor {
        let (srcIndices, outShape) = index.tensorSliceIndices(forShape: shape)
        return self.gather(outShape: outShape, srcIndices: srcIndices)
    }
}

func shapeProduct(_ shape: [Int]) -> Int {
    return shape.reduce(0, { x, y in x * y })
}
