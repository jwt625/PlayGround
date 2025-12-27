// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "PshMacOS",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "PshMacOS", targets: ["PshMacOS"])
    ],
    targets: [
        .executableTarget(
            name: "PshMacOS",
            dependencies: ["PshFFI"],
            linkerSettings: [
                .linkedLibrary("psh_ffi", .when(platforms: [.macOS])),
                .unsafeFlags(["-L../target/release"], .when(platforms: [.macOS]))
            ]
        ),
        .systemLibrary(
            name: "PshFFI",
            path: "Sources/PshFFI"
        )
    ]
)
