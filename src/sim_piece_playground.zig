const std = @import("std");
const ArrayList = std.ArrayList;
const print = std.debug.print;

// Import the sim_piece module - adjust path as needed
const sim_piece = @import("./functional/sim_piece.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Example 1: Simple linear data with some noise
    print("=== Example 1: Linear data with noise ===\n", .{});
    var linear_data = ArrayList(f64).init(allocator);
    defer linear_data.deinit();

    // Generate y = 2x + 1 with some noise
    for (0..20) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        const noise = (@as(f64, @floatFromInt(i % 3)) - 1.0) * 0.05; // Small noise
        try linear_data.append(2.0 * x + 1.0 + noise);
    }

    try testCompressionDecompression(allocator, linear_data.items, 0.1, "Linear with noise");

    // Example 2: Sine wave (should be harder to compress)
    print("\n=== Example 2: Sine wave ===\n", .{});
    var sine_data = ArrayList(f64).init(allocator);
    defer sine_data.deinit();

    for (0..50) |i| {
        const x: f64 = @as(f64, @floatFromInt(i)) * 0.2;
        try sine_data.append(@sin(x));
    }

    try testCompressionDecompression(allocator, sine_data.items, 0.1, "Sine wave");

    // Example 3: Step function (should compress very well)
    print("\n=== Example 3: Step function ===\n", .{});
    var step_data = ArrayList(f64).init(allocator);
    defer step_data.deinit();

    for (0..100) |i| {
        const value: f64 = if (i < 25) 1.0 else if (i < 50) 2.0 else if (i < 75) 1.5 else 3.0;
        try step_data.append(value);
    }

    try testCompressionDecompression(allocator, step_data.items, 0.05, "Step function");

    // Example 4: Your custom data
    print("\n=== Example 4: Custom data ===\n", .{});
    var custom_data = ArrayList(f64).init(allocator);
    defer custom_data.deinit();

    // Add your own data here!
    const values = [_]f64{ 1.0, 1.1, 1.05, 2.0, 2.1, 1.95, 3.0, 3.2, 2.9, 4.0, 4.1, 3.95 };
    try custom_data.appendSlice(&values);

    try testCompressionDecompression(allocator, custom_data.items, 0.2, "Custom data");
}

fn testCompressionDecompression(allocator: std.mem.Allocator, data: []const f64, error_bound: f32, description: []const u8) !void {
    print("Testing: {s}\n", .{description});
    print("Original data ({} points): ", .{data.len});
    for (data[0..@min(10, data.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (data.len > 10) print("...", .{});
    print("\n", .{});

    // Compress
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    try sim_piece.compressSimPiece(data, &compressed, allocator, error_bound);

    print("Compressed size: {} bytes\n", .{compressed.items.len});
    print("Compression ratio: {d:.2}:1\n", .{@as(f64, @floatFromInt(data.len * 8)) / @as(f64, @floatFromInt(compressed.items.len))});

    // Decompress
    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();

    try sim_piece.decompress(compressed.items, &decompressed, allocator);

    print("Decompressed data: ", .{});
    for (decompressed.items[0..@min(10, decompressed.items.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (decompressed.items.len > 10) print("...", .{});
    print("\n", .{});

    // Check error bounds
    var max_error: f64 = 0.0;
    var avg_error: f64 = 0.0;
    for (data, decompressed.items) |orig, decomp| {
        const diff = @abs(orig - decomp);
        max_error = @max(max_error, diff);
        avg_error += diff;
    }
    avg_error /= @as(f64, @floatFromInt(data.len));

    print("Max error: {d:.6} (bound: {d:.3})\n", .{ max_error, error_bound });
    print("Avg error: {d:.6}\n", .{avg_error});
    print("Within bounds: {}\n", .{max_error <= error_bound});
}
