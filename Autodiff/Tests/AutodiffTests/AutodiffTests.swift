import XCTest
@testable import Autodiff

final class AutodiffTests: XCTestCase {
    func testMSEGrad() throws {
        let x = Tensor(data: [1.0, 2.0, 3.0], shape: [3])
        let y = Tensor(data: [2.0, 0.0, -3.0], shape: [3])
        var xGrad: Tensor?
        let diff = x.onGrad({grad in xGrad = grad}) - y
        let sqDiff = diff * diff
        sqDiff.backward(grad: Tensor(onesLike: x))
        XCTAssertEqual(xGrad!.data, [-2, 4, 12])
    }

    func testMatrixVectorProduct() throws {
        let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [2, 3])
        let y = Tensor(data: [-1, -3, 2], shape: [3, 1])
        var xGrad: Tensor?
        var yGrad: Tensor?
        let xParam = x.onGrad { grad in xGrad = grad }
        let yParam = y.onGrad { grad in yGrad = grad }
        let product = matmul(xParam, yParam)
        XCTAssertEqual(product.data, [-1, -7])
        let outGrad = Tensor(onesLike: product)
        product.backward(grad: outGrad)
        XCTAssertEqual(xGrad!.data, [-1, -3, 2, -1, -3, 2])
        XCTAssertEqual(xGrad!.shape, x.shape)
        XCTAssertEqual(yGrad!.data, [5, 7, 9])
        XCTAssertEqual(yGrad!.shape, y.shape)
    }

    func testSlice() throws {
        let x = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [3, 2])
        XCTAssertEqual(x[1].data, [3, 4])
        XCTAssertEqual(x[1..<3].data, [3, 4, 5, 6])
        XCTAssertEqual(x[1...2].data, [3, 4, 5, 6])
        XCTAssertEqual(x[1..<2].data, [3, 4])
        XCTAssertEqual(x[(-2)..<(-1)].data, [3, 4])
        XCTAssertEqual(x[(-2)...(-1)].data, [3, 4, 5, 6])
        XCTAssertEqual(x[(-2)...].data, [3, 4, 5, 6])
        XCTAssertEqual(x[...(-2)].data, [1, 2, 3, 4])
        XCTAssertEqual(x[..<(-2)].data, [1, 2])

        let y = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [3, 1, 4])
        XCTAssertEqual(y[..., 0].shape, [3, 4])
        XCTAssertEqual(y[..., 0].data, y.data)
        XCTAssertEqual(y[0, 0, 0].shape, [])
        XCTAssertEqual(y[0, 0, 0].data, [1])
        XCTAssertEqual(y[0...1, ..., 3].data, [4, 8])
        XCTAssertEqual(y[0..<2, ..., 3].data, [4, 8])
        XCTAssertEqual(y[0...2, ..., 3].data, [4, 8, 12])
        XCTAssertEqual(y[0..<3, ..., 3].data, [4, 8, 12])
        XCTAssertEqual(y[0..., ..., 3].data, [4, 8, 12])
        XCTAssertEqual(y[1...2, ..., 2...3].data, [7, 8, 11, 12])
        XCTAssertEqual(y[0...2, ..., 2...3].data, [3, 4, 7, 8, 11, 12])
        XCTAssertEqual(y[0...2, 0..<1, 2...3].data, [3, 4, 7, 8, 11, 12])

        var yGrad: Tensor?
        let yParam = y.onGrad { grad in yGrad = grad }
        yParam[1...2, ..., 2...3].backward(grad: Tensor(data: [1, 2, 3, 4], shape: [2, 1, 2]))
        XCTAssertEqual(yGrad!.shape, y.shape)
        XCTAssertEqual(yGrad!.data, [0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4])
        
        let yParam1 = y.onGrad { grad in yGrad = grad }
        yParam1[..., 0, 3].backward(grad: Tensor(data: [1, 2, 3], shape: [3]))
        XCTAssertEqual(yGrad!.shape, y.shape)
        XCTAssertEqual(yGrad!.data, [0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3])
    }

    func testElemwise() throws {
        func testF(input: [Float], output: [Float], grad: [Float], _ op: (Tensor) -> Tensor) throws {
            var actualGrad: Tensor?
            let tensorIn = Tensor(data: input, shape: [input.count]) { g in
                actualGrad = g
            }
            let actualOut = op(tensorIn)
            try assertClose(actualOut, Tensor(data: output, shape: [output.count]))
            actualOut.backward(grad: Tensor(onesLike: actualOut))
            try assertClose(actualGrad!, Tensor(data: grad, shape: [output.count]))
        }

        try testF(
            input: [-1, -2, -3, 1, 2, 3],
            output: [1, 4, 9, 1, 4, 9],
            grad: [-2, -4, -6, 2, 4, 6]
        ) { $0.pow(2) }
        try testF(
            input: [0.01, 1, 2, 3],
            output: [0.1, 1.0, sqrt(2.0), sqrt(3.0)],
            grad: [5.0, 0.5, 0.3535533845424652, 0.28867512941360474]
        ) { $0.sqrt() }
        try testF(
            input: [-100, -2, 0, 1, 2, 3, 100],
            output: [-0.0, -0.04540228843688965, 0.0, 0.8411920070648193, 1.9545977115631104, 2.9963626861572266, 100.0],
            grad: [0.0, -0.08609922230243683, 0.5, 1.0829640626907349, 1.0860992670059204, 1.0115842819213867, 1.0]
        ) { $0.gelu() }
    }

    func testSum() throws {
        let input = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9], shape: [3, 3])
        var gradA: Tensor?
        let sumA = input.onGrad({ g in gradA = g }).sum(axis: 1)
        XCTAssertEqual(sumA.data, [6, 15, 24])
        sumA.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
        XCTAssertEqual(gradA!.data, [-1, -1, -1, -2, -2, -2, -3, -3, -3])

        var gradB: Tensor?
        let sumB = input.onGrad({ g in gradB = g }).sum(axis: 0)
        XCTAssertEqual(sumB.data, [12, 15, 18])
        sumB.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
        XCTAssertEqual(gradB!.data, [-1, -2, -3, -1, -2, -3, -1, -2, -3])

        let input1 = Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8], shape: [2, 2, 2])

        var gradC: Tensor?
        let sumC = input1.onGrad({ g in gradC = g }).sum(axis: 1)
        XCTAssertEqual(sumC.data, Array([1+3, 2+4, 5+7, 6+8].map { Float($0) }))
        sumC.backward(grad: Tensor(data: [-1, -2, -3, -4], shape: [2, 2]))
        XCTAssertEqual(gradC!.data, [-1, -2, -1, -2, -3, -4, -3, -4])

        var gradD: Tensor?
        let sumD = input1.onGrad({ g in gradD = g }).sum()
        XCTAssertEqual(sumD.data, [input1.data.sum()])
        sumD.backward(grad: Tensor(data: [-1], shape: []))
        XCTAssertEqual(gradD!.data, Array(repeating: Float(-1), count: 8))

        XCTAssertEqual(input1.sum(keepdims: true).shape, [1, 1, 1])
        XCTAssertEqual(input1.sum(axis: 0, keepdims: true).shape, [1, 2, 2])
        XCTAssertEqual(input1.sum(axis: 1, keepdims: true).shape, [2, 1, 2])
        XCTAssertEqual(input1.sum(axis: 2, keepdims: true).shape, [2, 2, 1])
    }

    func testMinMax() throws {
        let input = Tensor(data: [1, 10, 2, 7, 8, 9, 6, 4, 5], shape: [3, 3])
        var gradA: Tensor?
        let maxA = input.onGrad({ g in gradA = g }).max(axis: 1)
        XCTAssertEqual(maxA.data, [10, 9, 6])
        maxA.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
        XCTAssertEqual(gradA!.data, [0, -1, 0, 0, 0, -2, -3, 0, 0])

        var gradB: Tensor?
        let maxB = input.onGrad({ g in gradB = g }).max(axis: 0)
        XCTAssertEqual(maxB.data, [7, 10, 9])
        maxB.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
        XCTAssertEqual(gradB!.data, [0, -2, 0, -1, 0, -3, 0, 0, 0])

        var gradC: Tensor?
        let minC = input.onGrad({ g in gradC = g }).min(axis: 0)
        XCTAssertEqual(minC.data, [1, 4, 2])
        minC.backward(grad: Tensor(data: [-1, -2, -3], shape: [3]))
        XCTAssertEqual(gradC!.data, [-1, 0, -3, 0, 0, 0, 0, -2, 0])

        var gradD: Tensor?
        let maxD = input.onGrad({ g in gradD = g }).max()
        XCTAssertEqual(maxD.data, [10])
        maxD.backward(grad: Tensor(data: [-1], shape: []))
        XCTAssertEqual(gradD!.data, [0, -1, 0, 0, 0, 0, 0, 0, 0])

        XCTAssertEqual(input.max(keepdims: true).shape, [1, 1])
        XCTAssertEqual(input.max(axis: 0, keepdims: true).shape, [1, 3])
        XCTAssertEqual(input.max(axis: 1, keepdims: true).shape, [3, 1])
    }

    func testExpandAndRepeat() throws {
        let t1 = Tensor(data: [1, 2, 3, 4], shape: [2, 2])
        XCTAssertEqual(t1.reshape([1, 2, 1, 2]).expand(shape: [3, 7, 2, 5, 2]).shape, [3, 7, 2, 5, 2])

        let t2 = Tensor(data: [1, 2, 3, 4, 5, 6], shape: [1, 2, 1, 3, 1])
        XCTAssertEqual(t2.repeating(axis: 0, count: 2).data, t2.data + t2.data)
        XCTAssertEqual(t2.repeating(axis: 1, count: 2).data, t2.data + t2.data)
        XCTAssertEqual(t2.repeating(axis: 2, count: 2).data, [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
        XCTAssertEqual(t2.repeating(axis: 3, count: 2).data, [1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6])
        XCTAssertEqual(t2.repeating(axis: 4, count: 2).data, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])

        var grad: Tensor?
        let repeated = t2.onGrad({ g in grad = g }).repeating(axis: 2, count: 2)
        repeated.backward(grad: Tensor(data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], shape: [1, 2, 2, 3, 1]))
        XCTAssertEqual(grad!.data, [5, 7, 9, 17, 19, 21])
    }
}

func assertClose(_ x: Tensor, _ y: Tensor, atol: Float = 1e-4, rtol: Float = 1e-4) throws {
    XCTAssertEqual(x.shape, y.shape)
    var allGood = true
    for (a, b) in zip(x.data, y.data) {
        if abs(a-b)>atol && (b == 0 || abs(a/b-1) > rtol) {
            allGood = false
        }
    }
    XCTAssert(allGood, "tensors \(x.data) and \(y.data) are not equal")
}
