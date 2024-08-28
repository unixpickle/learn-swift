#if canImport(Accelerate)
import Accelerate
#endif

public extension Tensor {

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

    static func &* (lhs: Tensor, rhs: Tensor) -> Tensor {
        return tensorMatmul(lhs, rhs)
    }

}

private func tensorMatmul(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
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

    func cpuImplementation() {
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
    }

#if canImport(Accelerate)
    func wrapMatrix(
        _ data: Array<Float>,
        _ shape: [Int],
        _ f: (BNNSNDArrayDescriptor) -> Bool
    ) -> Bool {
        return data.withUnsafeBufferPointer { buffer in
            let arr = BNNSNDArrayDescriptor(
                flags: BNNSNDArrayFlags(0),
                layout: BNNSDataLayout2DFirstMajor,
                size: (shape[0], shape[1], 0, 0, 0, 0, 0, 0),
                stride: (0, 0, 0, 0, 0, 0, 0, 0),
                data: UnsafeMutableRawPointer(mutating: buffer.baseAddress!),
                data_type: .float,
                table_data: nil,
                table_data_type: .float,
                data_scale: 1,
                data_bias: 0
            )
            return f(arr)
        }
    }
    let success = wrapMatrix(lhs.data, lhs.shape) { lhsArray in
        wrapMatrix(rhs.data, rhs.shape) { rhsArray in
            wrapMatrix(data, shape) { outArray in
                do {
                    let size = BNNS.matrixMultiplicationWorkspaceSize(
                        inputA: lhsArray, transposed: false,
                        inputB: rhsArray, transposed: false,
                        output: outArray,
                        alpha: 1.0
                    )
                    guard let size = size else {
                        return false
                    }
                    var workspace = [UInt8](repeating: 0, count: size)
                    try workspace.withUnsafeMutableBufferPointer { workspacePtr in
                        try BNNS.applyMatrixMultiplication(
                            inputA: lhsArray, transposed: false,
                            inputB: rhsArray, transposed: false,
                            output: outArray,
                            alpha: 1.0,
                            workspace: UnsafeMutableRawBufferPointer(workspacePtr)
                        )
                    }
                } catch {
                    return false
                }
                return true
            }
        }
    }
    if !success {
        cpuImplementation()
    }
#else
    cpuImplementation()
#endif
    if !lhs.needsGrad && !rhs.needsGrad {
        return Tensor(data: data, shape: shape)
    } else {
        let lhsHandle = lhs.saveForBackward()
        let rhsHandle = rhs.saveForBackward()
        return Tensor(data: data, shape: shape) { grad in
            lhsHandle.backward(grad: tensorMatmul(grad, rhs.noGrad().transpose()))
            rhsHandle.backward(grad: tensorMatmul(lhs.noGrad().transpose(), grad))
        }
    }
}