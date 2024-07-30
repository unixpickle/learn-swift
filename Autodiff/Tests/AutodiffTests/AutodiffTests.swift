import XCTest
@testable import Autodiff

final class AutodiffTests: XCTestCase {
    func testMSEGrad() throws {
        let x = Tensor(data: [1.0, 2.0, 3.0], shape: [3])
        let y = Tensor(data: [2.0, 0.0, -3.0], shape: [3])
        var xGrad: Tensor?
        let diff = x.onGrad(action: {grad in xGrad = grad}) - y
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
}
