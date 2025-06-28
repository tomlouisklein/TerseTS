const std = @import("std");
const ArrayList = std.ArrayList;
const print = std.debug.print;
const sim_piece = @import("./functional/sim_piece.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test 1: Perfect linear data - should work even with error_bound = 0
    try testPerfectLinearData(allocator);
    
    // Test 2: Near-zero error bounds
    try testNearZeroErrorBounds(allocator);
    
    // Test 3: Floating point precision edge cases
    try testFloatingPointPrecision(allocator);
    
    // Test 4: Compression ratio vs error bound
    try testCompressionRatioVsError(allocator);
    
    // Test 5: Edge cases for quantization
    try testQuantizationEdgeCases(allocator);
}

fn testPerfectLinearData(allocator: std.mem.Allocator) !void {
    print("\n=== Test 1: Perfect Linear Data (testing error_bound = 0) ===\n", .{});
    
    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    
    // Generate perfectly linear data: y = 0.5x + 10
    for (0..20) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        try data.append(0.5 * x + 10.0);
    }
    
    // Test with different error bounds including 0
    const error_bounds = [_]f32{ 0.0, 0.000001, 0.0001, 0.01, 0.1 };
    
    for (error_bounds) |error_bound| {
        print("\n--- Error bound: {d:.6} ---\n", .{error_bound});
        
        var compressed = ArrayList(u8).init(allocator);
        defer compressed.deinit();
        
        // Try to compress - this might fail for error_bound = 0
        sim_piece.compressSimPiece(data.items, &compressed, allocator, error_bound) catch |err| {
            print("Compression failed with error: {}\n", .{err});
            continue;
        };
        
        print("Compressed size: {} bytes\n", .{compressed.items.len});
        print("Compression ratio: {d:.2}:1\n", .{
            @as(f64, @floatFromInt(data.items.len * 8)) / @as(f64, @floatFromInt(compressed.items.len))
        });
        
        // Verify decompression
        var decompressed = ArrayList(f64).init(allocator);
        defer decompressed.deinit();
        
        try sim_piece.decompress(compressed.items, &decompressed, allocator);
        
        var max_error: f64 = 0.0;
        for (data.items, decompressed.items) |orig, decomp| {
            max_error = @max(max_error, @abs(orig - decomp));
        }
        print("Max reconstruction error: {d:.9}\n", .{max_error});
    }
}

fn testNearZeroErrorBounds(allocator: std.mem.Allocator) !void {
    print("\n=== Test 2: Near-Zero Error Bounds ===\n", .{});
    
    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    
    // Generate data with tiny variations
    for (0..30) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        const tiny_noise = @sin(x) * 0.00001; // Very small noise
        try data.append(x + tiny_noise);
    }
    
    // Test progressively smaller error bounds
    var error_bound: f32 = 1.0;
    while (error_bound > 1e-10) : (error_bound *= 0.1) {
        print("\n--- Testing error_bound: {e} ---\n", .{error_bound});
        
        var compressed = ArrayList(u8).init(allocator);
        defer compressed.deinit();
        
        sim_piece.compressSimPiece(data.items, &compressed, allocator, error_bound) catch |err| {
            print("Failed at error_bound {e}: {}\n", .{ error_bound, err });
            break;
        };
        
        var decompressed = ArrayList(f64).init(allocator);
        defer decompressed.deinit();
        try sim_piece.decompress(compressed.items, &decompressed, allocator);
        
        var segments_count: usize = 0;
        // Rough estimate of segments based on compressed size
        segments_count = compressed.items.len / 16; // Approximate
        
        print("Compressed size: {} bytes, ~{} segments\n", .{ compressed.items.len, segments_count });
        print("Compression ratio: {d:.2}:1\n", .{
            @as(f64, @floatFromInt(data.items.len * 8)) / @as(f64, @floatFromInt(compressed.items.len))
        });
    }
}

fn testFloatingPointPrecision(allocator: std.mem.Allocator) !void {
    print("\n=== Test 3: Floating Point Precision Edge Cases ===\n", .{});
    
    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    
    // Generate data that looks linear but has floating point precision issues
    const base: f64 = 1.0 / 3.0; // 0.333333...
    for (0..20) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        try data.append(base * x);
    }
    
    print("Data values (first 5): ", .{});
    for (data.items[0..5]) |val| {
        print("{d:.15} ", .{val});
    }
    print("\n", .{});
    
    // Test with very small error bounds
    const error_bounds = [_]f32{ 0.0, 1e-15, 1e-10, 1e-7, 1e-5 };
    
    for (error_bounds) |error_bound| {
        print("\n--- Error bound: {e} ---\n", .{error_bound});
        
        var compressed = ArrayList(u8).init(allocator);
        defer compressed.deinit();
        
        sim_piece.compressSimPiece(data.items, &compressed, allocator, error_bound) catch |err| {
            print("Compression failed: {}\n", .{err});
            continue;
        };
        
        print("Compressed size: {} bytes\n", .{compressed.items.len});
    }
}

fn testCompressionRatioVsError(allocator: std.mem.Allocator) !void {
    print("\n=== Test 4: Compression Ratio vs Error Bound ===\n", .{});
    
    var data = ArrayList(f64).init(allocator);
    defer data.deinit();
    
    // Generate realistic sensor data with noise
    for (0..100) |i| {
        const t: f64 = @as(f64, @floatFromInt(i)) * 0.1;
        const signal = @sin(t) + 0.5 * @sin(3 * t);
        const noise = (@as(f64, @floatFromInt(i % 7)) - 3.0) * 0.01;
        try data.append(signal + noise);
    }
    
    print("Testing compression efficiency as error_bound approaches zero:\n", .{});
    print("Error Bound | Compressed Size | Ratio | Max Error\n", .{});
    print("----------- | --------------- | ----- | ---------\n", .{});
    
    const error_bounds = [_]f32{ 1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001 };
    
    for (error_bounds) |error_bound| {
        var compressed = ArrayList(u8).init(allocator);
        defer compressed.deinit();
        
        sim_piece.compressSimPiece(data.items, &compressed, allocator, error_bound) catch {
            print("{d:11.6} | FAILED          |       |          \n", .{error_bound});
            continue;
        };
        
        var decompressed = ArrayList(f64).init(allocator);
        defer decompressed.deinit();
        try sim_piece.decompress(compressed.items, &decompressed, allocator);
        
        var max_error: f64 = 0.0;
        for (data.items, decompressed.items) |orig, decomp| {
            max_error = @max(max_error, @abs(orig - decomp));
        }
        
        const ratio = @as(f64, @floatFromInt(data.items.len * 8)) / @as(f64, @floatFromInt(compressed.items.len));
        
        print("{d:11.6} | {d:15} | {d:5.1} | {d:9.6}\n", .{
            error_bound,
            compressed.items.len,
            ratio,
            max_error,
        });
    }
}

fn testQuantizationEdgeCases(allocator: std.mem.Allocator) !void {
    print("\n=== Test 5: Quantization Edge Cases ===\n", .{});
    
    // Test case 1: Values that would quantize to the same value with error_bound = 0
    print("\n--- Test 5a: Identical starting points ---\n", .{});
    var data1 = ArrayList(f64).init(allocator);
    defer data1.deinit();
    
    // Multiple segments starting at exactly 1.0
    try data1.append(1.0);
    try data1.append(2.0);
    try data1.append(1.0); // Back to 1.0
    try data1.append(3.0);
    try data1.append(1.0); // Back to 1.0 again
    try data1.append(4.0);
    
    try testCompressionDecompression(allocator, data1.items, 0.0, "Repeated starting points");
    
    // Test case 2: Values that differ by less than machine epsilon
    print("\n--- Test 5b: Machine epsilon differences ---\n", .{});
    var data2 = ArrayList(f64).init(allocator);
    defer data2.deinit();
    
    const epsilon = std.math.floatEps(f64);
    try data2.append(1.0);
    try data2.append(1.0 + epsilon);
    try data2.append(1.0 + 2 * epsilon);
    try data2.append(1.0 + 3 * epsilon);
    
    print("Epsilon value: {e}\n", .{epsilon});
    print("Data: ", .{});
    for (data2.items) |val| {
        print("{d:.20} ", .{val});
    }
    print("\n", .{});
    
    const tiny_bounds = [_]f32{ 0.0, @floatCast(epsilon), @floatCast(epsilon * 10) };
    for (tiny_bounds) |bound| {
        print("\nTesting with bound {e}:\n", .{bound});
        try testCompressionDecompression(allocator, data2.items, bound, "Epsilon differences");
    }
}

// Helper function from original code
fn testCompressionDecompression(allocator: std.mem.Allocator, data: []const f64, error_bound: f32, description: []const u8) !void {
    print("Testing: {s}\n", .{description});
    print("Original data ({} points): ", .{data.len});
    for (data[0..@min(10, data.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (data.len > 10) print("...", .{});
    print("\n", .{});
    
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();
    
    sim_piece.compressSimPiece(data, &compressed, allocator, error_bound) catch |err| {
        print("Compression failed: {}\n", .{err});
        return;
    };
    
    print("Compressed size: {} bytes\n", .{compressed.items.len});
    print("Compression ratio: {d:.2}:1\n", .{
        @as(f64, @floatFromInt(data.len * 8)) / @as(f64, @floatFromInt(compressed.items.len))
    });
    
    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();
    
    try sim_piece.decompress(compressed.items, &decompressed, allocator);
    
    print("Decompressed data: ", .{});
    for (decompressed.items[0..@min(10, decompressed.items.len)]) |val| {
        print("{d:.3} ", .{val});
    }
    if (decompressed.items.len > 10) print("...", .{});
    print("\n", .{});
    
    var max_error: f64 = 0.0;
    var avg_error: f64 = 0.0;
    for (data, decompressed.items) |orig, decomp| {
        const diff = @abs(orig - decomp);
        max_error = @max(max_error, diff);
        avg_error += diff;
    }
    avg_error /= @as(f64, @floatFromInt(data.len));
    
    print("Max error: {d:.9} (bound: {d:.6})\n", .{ max_error, error_bound });
    print("Avg error: {d:.9}\n", .{avg_error});
    print("Within bounds: {}\n", .{max_error <= error_bound});
}
