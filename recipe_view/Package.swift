// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "recipe_view",
    dependencies: [
        .package(url: "https://github.com/stackotter/swift-cross-ui", branch: "main"),
        .package(url: "https://github.com/swift-server/async-http-client.git", from: "1.9.0")
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .executableTarget(
            name: "recipe_view",
            dependencies: [
                .product(name: "SwiftCrossUI", package: "swift-cross-ui"),
                .product(name: "GtkBackend", package: "swift-cross-ui"),
                .product(name: "AsyncHTTPClient", package: "async-http-client")
            ]
        ),
    ]
)
