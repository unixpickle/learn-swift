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
}
