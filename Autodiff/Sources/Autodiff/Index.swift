protocol TensorIndex {
    var minTensorSliceDims: Int { get }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int])
}

extension Sequence where Element: Numeric {
    func product() -> Element {
        reduce(Element(exactly: 1)!, { x, y in x * y })
    }

    func sum() -> Element {
        reduce(Element(exactly: 0)!, { x, y in x + y })
    }
}

func allIndices<S: Sequence>(forShape inShape: S) -> [Int] where S.Element == Int {
    Array(0..<inShape.product())
}

func allIndices(forShape inShape: [Int]) -> [Int] {
    Array(0..<inShape.product())
}

extension Int: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let idx = self < 0 ? inShape[0] + self : self
        let innerIndices = allIndices(forShape: inShape[1...])
        let offset = idx*innerIndices.count
        return (Array(innerIndices.map({$0 + offset})), Array(inShape[1...]))
    }
}

extension Range<Int>: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let start = self.lowerBound < 0 ? inShape[0] + self.lowerBound : self.lowerBound
        let end = self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound
        assert(end >= start, "end (\(end)) must be >= start (\(start))")
        var result: [Int] = Array()
        for i in start..<end {
            result += i.tensorSliceIndices(forShape: inShape).0
        }
        return (result, [end - start] + inShape[1...])
    }
}

extension ClosedRange<Int>: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let end = 1 + (self.upperBound < 0 ? inShape[0] + self.upperBound : self.upperBound)
        return (self.lowerBound..<end).tensorSliceIndices(forShape: inShape)
    }
}

extension UnboundedRange_: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        return (allIndices(forShape: inShape), inShape)
    }
}

extension PartialRangeFrom<Int>: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        return (self.lowerBound..<inShape[0]).tensorSliceIndices(forShape: inShape)
    }
}

extension PartialRangeUpTo<Int>: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
        return (0..<end).tensorSliceIndices(forShape: inShape)
    }
}

extension PartialRangeThrough<Int>: TensorIndex {
    var minTensorSliceDims: Int { 1 }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count > 0)
        let end = (self.upperBound > 0 ? self.upperBound : self.upperBound + inShape[0])
        return (0...end).tensorSliceIndices(forShape: inShape)
    }
}

extension Array: TensorIndex where Element: TensorIndex {
    var minTensorSliceDims: Int { self.map({ $0.minTensorSliceDims }).sum() }

    func tensorSliceIndices(forShape inShape: [Int]) -> ([Int], [Int]) {
        assert(inShape.count >= self.minTensorSliceDims)
        let a = Array(self)
        switch a.count {
        case 0:
            return (allIndices(forShape: inShape), inShape)
        case 1:
            return a[0].tensorSliceIndices(forShape: inShape)
        default:
            let currentShape: [Int] = Array<Int>(inShape[...a[0].minTensorSliceDims])
            let laterShape: [Int] = Array<Int>(inShape[a[0].minTensorSliceDims...])
            let (subIndices, subShape) = Array(a[1...]).tensorSliceIndices(forShape: laterShape)
            let (firstIndices, firstShape) = a[0].tensorSliceIndices(forShape: currentShape)
            var result: [Int] = []
            let innerSize = laterShape.product()
            for firstIdx in firstIndices {
                let offset = firstIdx * innerSize
                result += subIndices.map({$0 + offset})
            }
            return (result, firstShape + subShape)
        }
    }
}

extension Tensor {
    func scatter(outShape: [Int], dstIndices: [Int]) -> Tensor {
        assert(dstIndices.count == data.count)
        var newData: [Float] = Array(repeating: 0, count: outShape.product())
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
        var newData: [Float] = Array(repeating: 0, count: outShape.product())
        assert(
            srcIndices.count == newData.count,
            "expected indices count \(srcIndices.count) to equal to size of shape \(outShape)"
        )
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

    subscript<T>(index: T...) -> Tensor where T: TensorIndex {
        let (srcIndices, outShape) = Array(index).tensorSliceIndices(forShape: shape)
        return self.gather(outShape: outShape, srcIndices: srcIndices)
    }
}
