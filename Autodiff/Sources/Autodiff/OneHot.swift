extension Tensor {

    convenience init(oneHot: Int, count: Int) {
        assert(oneHot >= 0 && oneHot < count, "oneHot \(oneHot) out of range [0, \(count))")
        var data = Array(repeating: Float(0), count: count)
        data[oneHot] = 1
        self.init(data: data, shape: [count])
    }

    convenience init(oneHot: [Int], count: Int) {
        var data = Array(repeating: Float(0), count: count * oneHot.count)
        for (i, idx) in oneHot.enumerated() {
            assert(idx >= 0 && idx < count, "oneHot index \(idx) out of range [0, \(count))")
            data[i * count + idx] = 1
        }
        self.init(data: data, shape: [oneHot.count, count])
    }

}