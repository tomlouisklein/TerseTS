const std = @import("std");
const ArrayList = std.ArrayList;
const print = std.debug.print;

// Import the mixed_type_pla module - adjust path as needed
const mixed_type_pla = @import("./functional/mixed_type_PLA.zig");

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

    // Example 4: Data with trend (Mixed-type PLA should handle this well)
    print("\n=== Example 4: Data with trend ===\n", .{});
    var trend_data = ArrayList(f64).init(allocator);
    defer trend_data.deinit();

    for (0..50) |i| {
        const x: f64 = @as(f64, @floatFromInt(i));
        const trend = x * 0.5;
        const seasonal = 3.0 * @sin(x * 0.3);
        const noise = (@as(f64, @floatFromInt(i % 5)) - 2.0) * 0.1;
        try trend_data.append(trend + seasonal + noise);
    }

    try testCompressionDecompression(allocator, trend_data.items, 0.2, "Data with trend");

    // Example 5: Piecewise data with different slopes
    print("\n=== Example 5: Piecewise linear segments ===\n", .{});
    var piecewise_data = ArrayList(f64).init(allocator);
    defer piecewise_data.deinit();

    // Segment 1: slope = 1
    for (0..10) |i| {
        try piecewise_data.append(@as(f64, @floatFromInt(i)));
    }
    // Segment 2: slope = -0.5
    for (10..20) |i| {
        try piecewise_data.append(10.0 - 0.5 * @as(f64, @floatFromInt(i - 10)));
    }
    // Segment 3: slope = 2
    for (20..30) |i| {
        try piecewise_data.append(5.0 + 2.0 * @as(f64, @floatFromInt(i - 20)));
    }

    try testCompressionDecompression(allocator, piecewise_data.items, 0.1, "Piecewise linear");

    // Example 6: Constant segments (should group well with mixed knots)
    print("\n=== Example 6: Constant segments ===\n", .{});
    var constant_data = ArrayList(f64).init(allocator);
    defer constant_data.deinit();

    // Multiple segments with the same value - perfect for mixed-type PLA
    for (0..5) |_| try constant_data.append(1.0);
    for (0..5) |_| try constant_data.append(2.0);
    for (0..5) |_| try constant_data.append(1.0); // Same as first segment
    for (0..5) |_| try constant_data.append(3.0);
    for (0..5) |_| try constant_data.append(2.0); // Same as second segment

    try testCompressionDecompression(allocator, constant_data.items, 0.01, "Constant segments");

    // Example 7: Small dataset
    print("\n=== Example 7: Small dataset ===\n", .{});
    var small_data = ArrayList(f64).init(allocator);
    defer small_data.deinit();
    const small_values = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try small_data.appendSlice(&small_values);

    try testCompressionDecompression(allocator, small_data.items, 0.1, "Small dataset");

    // Example 8: Mixed knot showcase (perfect for demonstrating mixed-type PLA advantage)
    print("\n=== Example 8: Mixed knot showcase ===\n", .{});
    var mixed_knot_data = ArrayList(f64).init(allocator);
    defer mixed_knot_data.deinit();

    // Create data that benefits from both joint and disjoint knots
    // First segment: linear increase
    for (0..10) |i| {
        try mixed_knot_data.append(@as(f64, @floatFromInt(i)) * 0.5);
    }
    // Jump (disjoint knot beneficial)
    for (10..15) |i| {
        try mixed_knot_data.append(10.0 + @as(f64, @floatFromInt(i - 10)) * 0.3);
    }
    // Continuous segment (joint knot beneficial)
    for (15..25) |i| {
        try mixed_knot_data.append(11.5 + @as(f64, @floatFromInt(i - 15)) * 0.8);
    }

    try testCompressionDecompression(allocator, mixed_knot_data.items, 0.1, "Mixed knot showcase");

    // Example 9: Zigzag pattern
    print("\n=== Example 9: Zigzag pattern ===\n", .{});
    var zigzag_data = ArrayList(f64).init(allocator);
    defer zigzag_data.deinit();

    for (0..30) |i| {
        const value = if (i % 2 == 0) @as(f64, @floatFromInt(i / 2)) else @as(f64, @floatFromInt(i / 2)) + 0.5;
        try zigzag_data.append(value);
    }

    try testCompressionDecompression(allocator, zigzag_data.items, 0.15, "Zigzag pattern");
}

fn testCompressionDecompression(
    allocator: std.mem.Allocator,
    data: []const f64,
    error_bound: f32,
    description: []const u8,
) !void {
    print("Testing: {s}\n", .{description});
    print("Original data ({} points): ", .{data.len});
    for (data[0..data.len]) |val| {
        print("{d:.3} ", .{val});
    }
    print("\n", .{});

    // Compress using Mixed-type PLA
    var compressed = ArrayList(u8).init(allocator);
    defer compressed.deinit();

    try mixed_type_pla.compress(data, &compressed, allocator, error_bound);

    print("Compressed size: {} bytes\n", .{compressed.items.len});
    print("Compression ratio: {d:.2}:1\n", .{
        @as(f64, @floatFromInt(data.len * 8)) / @as(f64, @floatFromInt(compressed.items.len)),
    });

    // Try to parse basic structure info if possible
    if (compressed.items.len >= @sizeOf(usize)) {
        // The exact structure depends on the implementation, but we can show basic info
        print("  - Compressed representation uses mixed joint/disjoint knots\n", .{});
    }

    // Decompress
    var decompressed = ArrayList(f64).init(allocator);
    defer decompressed.deinit();

    try mixed_type_pla.decompress(compressed.items, &decompressed, allocator);

    print("Decompressed data: ", .{});
    for (decompressed.items[0..decompressed.items.len]) |val| {
        print("{d:.3} ", .{val});
    }
    print("\n", .{});

    // Verify length
    if (data.len != decompressed.items.len) {
        print("ERROR: Length mismatch! Original: {}, Decompressed: {}\n", .{ data.len, decompressed.items.len });
        return;
    }

    // Check error bounds
    var max_error: f64 = 0.0;
    var avg_error: f64 = 0.0;
    var error_positions = ArrayList(usize).init(allocator);
    defer error_positions.deinit();

    for (data, decompressed.items, 0..) |orig, decomp, i| {
        const diff = @abs(orig - decomp);
        max_error = @max(max_error, diff);
        avg_error += diff;

        if (diff > error_bound) {
            try error_positions.append(i);
        }
    }
    avg_error /= @as(f64, @floatFromInt(data.len));

    print("Max error: {d:.6} (bound: {d:.3})\n", .{ max_error, error_bound });
    print("Avg error: {d:.6}\n", .{avg_error});
    print("Within bounds: {}\n", .{max_error <= error_bound});

    if (error_positions.items.len > 0) {
        print("ERROR: Found {} positions exceeding error bound:\n", .{error_positions.items.len});
        for (error_positions.items[0..@min(5, error_positions.items.len)]) |pos| {
            print("  Position {}: orig={d:.6}, decomp={d:.6}, error={d:.6}\n", .{
                pos,
                data[pos],
                decompressed.items[pos],
                @abs(data[pos] - decompressed.items[pos]),
            });
        }
        if (error_positions.items.len > 5) print("  ... and {} more\n", .{error_positions.items.len - 5});
    }

    print("Status: {s}\n", .{if (max_error <= error_bound) "PASS ✓" else "FAIL ✗"});

    // Additional analysis for Mixed-type PLA
    analyzeCompressionCharacteristics(data, error_bound, description);
}

fn analyzeCompressionCharacteristics(data: []const f64, error_bound: f32, description: []const u8) void {
    _ = error_bound;
    print("Analysis for {s}:\n", .{description});

    // Calculate data characteristics that might influence mixed-type PLA performance
    var total_variation: f64 = 0.0;
    var max_jump: f64 = 0.0;
    var avg_slope_change: f64 = 0.0;
    var slope_changes: usize = 0;

    if (data.len >= 2) {
        for (1..data.len) |i| {
            const diff = @abs(data[i] - data[i - 1]);
            total_variation += diff;
            max_jump = @max(max_jump, diff);

            if (i >= 2) {
                const slope1 = data[i - 1] - data[i - 2];
                const slope2 = data[i] - data[i - 1];
                const slope_change = @abs(slope2 - slope1);
                avg_slope_change += slope_change;
                slope_changes += 1;
            }
        }

        if (slope_changes > 0) {
            avg_slope_change /= @as(f64, @floatFromInt(slope_changes));
        }
    }

    print("  - Total variation: {d:.3}\n", .{total_variation});
    print("  - Max jump: {d:.3}\n", .{max_jump});
    print("  - Avg slope change: {d:.3}\n", .{avg_slope_change});

    // Heuristic assessment for mixed-type PLA suitability
    if (max_jump > 2.0 * (total_variation / @as(f64, @floatFromInt(data.len)))) {
        print("  - Assessment: Good candidate for disjoint knots (large jumps detected)\n", .{});
    } else if (avg_slope_change < 0.5) {
        print("  - Assessment: Good candidate for joint knots (smooth transitions)\n", .{});
    } else {
        print("  - Assessment: Excellent candidate for mixed-type PLA (varying characteristics)\n", .{});
    }

    print("\n", .{});
}
