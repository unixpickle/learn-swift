extension Tensor {
    func transpose() -> Tensor {
        assert(shape.count == 2, "can only transpose two-dimensional tensors")
        let newShape = [shape[1], shape[0]]
        var newData = Array(repeating: Float(0), count: data.count)
        var dstIdx = 0
        for i in 0..<shape[1] {
            for j in 0..<shape[0] {
                newData[dstIdx] = data[i + j*shape[1]]
                dstIdx += 1
            }
        }
        if !needsGrad {
            return Tensor(data: newData, shape: newShape)
        } else {
            let handle = saveForBackward()
            return Tensor(data: newData, shape: newShape) { grad in
                handle.backward(grad: grad.transpose())
            }
        }
    }
}

func matmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
    assert(
        lhs.shape.count == 2 && rhs.shape.count == 2,
        "matrices must be two-dimensional: lhs.shape=\(lhs.shape) rhs.shape=\(rhs.shape)"
    )
    assert(
        lhs.shape[1] == rhs.shape[0],
        "matmul shape mismatch: lhs.shape=\(lhs.shape) rhs.shape=\(rhs.shape)"
    )
    let shape = [lhs.shape[0], rhs.shape[1]]
    var data = Array(repeating: Float(0), count: lhs.shape[0] * rhs.shape[1])
    for outRow in 0..<shape[0] {
        for outCol in 0..<shape[1] {
            var sum: Float = 0.0
            for k in 0..<lhs.shape[1] {
                let lhsIdx = outRow*lhs.shape[1] + k
                let rhsIdx = k*rhs.shape[1] + outCol
                sum += lhs.data[lhsIdx] * rhs.data[rhsIdx]
            }
            data[outRow*shape[1] + outCol] = sum
        }
    }
    if !lhs.needsGrad && !rhs.needsGrad {
        return Tensor(data: data, shape: shape)
    } else {
        let lhsHandle = lhs.saveForBackward()
        let rhsHandle = rhs.saveForBackward()
        return Tensor(data: data, shape: shape) { grad in
            lhsHandle.backward(grad: matmul(grad, rhs.noGrad().transpose()))
            rhsHandle.backward(grad: matmul(lhs.noGrad().transpose(), grad))
        }
    }
}