const std = @import("std");
const testing = std.testing;
const print = std.debug.print;
const ArrayList = std.ArrayList;
const math = std.math;

// Import your module - adjust the path as needed
const mix_piece = @import("functional/sim_piece.zig"); // Adjust this import path

// Test data generators
fn generateLinearData(allocator: std.mem.Allocator, start: f64, end: f64, num_points: usize) ![]f64 {
    var data = try allocator.alloc(f64, num_points);
    const step = (end - start) / @as(f64, @floatFromInt(num_points - 1));

    for (0..num_points) |i| {
        data[i] = start + step * @as(f64, @floatFromInt(i));
    }

    return data;
}

fn generateSineWave(allocator: std.mem.Allocator, amplitude: f64, frequency: f64, num_points: usize) ![]f64 {
    var data = try allocator.alloc(f64, num_points);

    for (0..num_points) |i| {
        const t = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_points - 1)) * 2.0 * math.pi * frequency;
        data[i] = amplitude * @sin(t);
    }

    return data;
}

fn generateConstantData(allocator: std.mem.Allocator, value: f64, num_points: usize) ![]f64 {
    var data = try allocator.alloc(f64, num_points);

    for (0..num_points) |i| {
        data[i] = value;
    }

    return data;
}

fn generateStepData(allocator: std.mem.Allocator, step_size: f64, num_points: usize) ![]f64 {
    var data = try allocator.alloc(f64, num_points);

    for (0..num_points) |i| {
        data[i] = @as(f64, @floatFromInt(i / 10)) * step_size; // Step every 10 points
    }

    return data;
}

fn generateNoisyLinearData(allocator: std.mem.Allocator, start: f64, end: f64, noise_amplitude: f64, num_points: usize) ![]f64 {
    var data = try allocator.alloc(f64, num_points);
    const step = (end - start) / @as(f64, @floatFromInt(num_points - 1));

    var prng = std.Random.DefaultPrng.init(12345); // Fixed seed for reproducible tests
    var random = prng.random();

    for (0..num_points) |i| {
        const linear_value = start + step * @as(f64, @floatFromInt(i));
        const noise = (random.float(f64) - 0.5) * 2.0 * noise_amplitude;
        data[i] = linear_value + noise;
    }

    return data;
}

// Helper function to print segment metadata
fn printSegmentMetadata(segments: []const mix_piece.SegmentMetadata) void {
    print("Number of segments: {}\n", .{segments.len});
    for (segments, 0..) |segment, i| {
        print("Segment {}: start_time={}, interception={d:.4}, upper_slope={d:.4}, lower_slope={d:.4}\n", .{ i, segment.start_time, segment.interception, segment.upper_bound_slope, segment.lower_bound_slope });
    }
}

// Helper function to validate segment bounds
fn validateSegmentBounds(segment: mix_piece.SegmentMetadata) bool {
    return segment.lower_bound_slope <= segment.upper_bound_slope;
}

// Helper function to check if a point is within segment bounds
fn isPointWithinBounds(segment: mix_piece.SegmentMetadata, time_offset: f64, value: f64, error_bound: f32) bool {
    const upper_line = segment.upper_bound_slope * time_offset + segment.interception;
    const lower_line = segment.lower_bound_slope * time_offset + segment.interception;

    return (value >= lower_line - error_bound) and (value <= upper_line + error_bound);
}

test "computeSegmentsMetadata - linear data with small error bound" {
    print("\n=== Test 1: Linear data with small error bound ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate linear data from 0 to 10
    const data = try generateLinearData(allocator, 0.0, 10.0, 11);
    defer allocator.free(data);

    print("Input data: ", .{});
    for (data) |value| {
        print("{d:.2} ", .{value});
    }
    print("\n", .{});

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 0.1);

    printSegmentMetadata(segments_metadata.items);

    // Validate results
    try testing.expect(segments_metadata.items.len > 0);
    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
    }

    print("✓ Linear data test passed\n", .{});
}

test "computeSegmentsMetadata - constant data" {
    print("\n=== Test 2: Constant data ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try generateConstantData(allocator, 5.0, 20);
    defer allocator.free(data);

    print("Input data: constant value 5.0 for 20 points\n", .{});

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 0.5);

    printSegmentMetadata(segments_metadata.items);

    // For constant data, we should get very few segments (ideally 1)
    try testing.expect(segments_metadata.items.len <= 3);

    // All segments should have slopes close to 0
    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
        try testing.expect(@abs(segment.upper_bound_slope) < 1.0);
        try testing.expect(@abs(segment.lower_bound_slope) < 1.0);
    }

    print("✓ Constant data test passed\n", .{});
}

test "computeSegmentsMetadata - sine wave" {
    print("\n=== Test 3: Sine wave ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try generateSineWave(allocator, 2.0, 1.0, 50);
    defer allocator.free(data);

    print("Input data: sine wave with amplitude 2.0, 50 points\n", .{});

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 0.2);

    printSegmentMetadata(segments_metadata.items);

    // Sine wave should require multiple segments
    try testing.expect(segments_metadata.items.len > 5);

    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
    }

    print("✓ Sine wave test passed\n", .{});
}

test "computeSegmentsMetadata - step function" {
    print("\n=== Test 4: Step function ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try generateStepData(allocator, 1.0, 50);
    defer allocator.free(data);

    print("Input data: step function, 50 points\n", .{});

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 0.1);

    printSegmentMetadata(segments_metadata.items);

    try testing.expect(segments_metadata.items.len > 0);

    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
    }

    print("✓ Step function test passed\n", .{});
}

test "computeSegmentsMetadata - noisy linear data" {
    print("\n=== Test 5: Noisy linear data ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try generateNoisyLinearData(allocator, 0.0, 10.0, 0.5, 30);
    defer allocator.free(data);

    print("Input data: noisy linear trend from 0 to 10 with noise ±0.5\n", .{});

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 1.0);

    printSegmentMetadata(segments_metadata.items);

    try testing.expect(segments_metadata.items.len > 0);

    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
    }

    print("✓ Noisy linear data test passed\n", .{});
}

test "computeSegmentsMetadata - single point" {
    print("\n=== Test 6: Single point ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f64{5.0};

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(&data, &segments_metadata, 0.1);

    printSegmentMetadata(segments_metadata.items);

    // Single point should result in no segments (since we need at least 2 points)
    try testing.expect(segments_metadata.items.len == 0);

    print("✓ Single point test passed\n", .{});
}

test "computeSegmentsMetadata - two points" {
    print("\n=== Test 7: Two points ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f64{ 1.0, 3.0 };

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    try mix_piece.computeSegmentsMetadata(&data, &segments_metadata, 0.1);

    printSegmentMetadata(segments_metadata.items);

    // Two points should result in exactly one segment
    try testing.expect(segments_metadata.items.len == 1);

    const segment = segments_metadata.items[0];
    try testing.expect(validateSegmentBounds(segment));
    try testing.expect(segment.start_time == 0);

    print("✓ Two points test passed\n", .{});
}

test "computeSegmentsMetadata - different error bounds" {
    print("\n=== Test 8: Different error bounds ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = try generateSineWave(allocator, 1.0, 2.0, 30);
    defer allocator.free(data);

    const error_bounds = [_]f32{ 0.05, 0.1, 0.2, 0.5, 1.0 };

    for (error_bounds) |error_bound| {
        print("Testing with error bound: {d:.2}\n", .{error_bound});

        var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
        defer segments_metadata.deinit();

        try mix_piece.computeSegmentsMetadata(data, &segments_metadata, error_bound);

        print("  Segments: {}\n", .{segments_metadata.items.len});

        try testing.expect(segments_metadata.items.len > 0);

        for (segments_metadata.items) |segment| {
            try testing.expect(validateSegmentBounds(segment));
        }
    }

    print("✓ Different error bounds test passed\n", .{});
}

test "computeSegmentsMetadata - invalid input (NaN)" {
    print("\n=== Test 9: Invalid input (NaN) ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f64{ 1.0, 2.0, std.math.nan(f64), 4.0 };

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    const result = mix_piece.computeSegmentsMetadata(&data, &segments_metadata, 0.1);

    // Should return an error for NaN input
    try testing.expectError(mix_piece.Error.UnsupportedInput, result);

    print("✓ NaN input test passed (correctly rejected)\n", .{});
}

test "computeSegmentsMetadata - invalid input (infinity)" {
    print("\n=== Test 10: Invalid input (infinity) ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const data = [_]f64{ 1.0, std.math.inf(f64), 3.0 };

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    const result = mix_piece.computeSegmentsMetadata(&data, &segments_metadata, 0.1);

    // Should return an error for infinity input
    try testing.expectError(mix_piece.Error.UnsupportedInput, result);

    print("✓ Infinity input test passed (correctly rejected)\n", .{});
}

test "computeSegmentsMetadata - large dataset performance" {
    print("\n=== Test 11: Large dataset performance ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Generate a larger dataset
    const data = try generateNoisyLinearData(allocator, 0.0, 100.0, 1.0, 1000);
    defer allocator.free(data);

    var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
    defer segments_metadata.deinit();

    const start_time = std.time.milliTimestamp();
    try mix_piece.computeSegmentsMetadata(data, &segments_metadata, 2.0);
    const end_time = std.time.milliTimestamp();

    print("Processed 1000 points in {} ms\n", .{end_time - start_time});
    print("Generated {} segments\n", .{segments_metadata.items.len});

    try testing.expect(segments_metadata.items.len > 0);
    try testing.expect(segments_metadata.items.len < 500); // Should compress significantly

    for (segments_metadata.items) |segment| {
        try testing.expect(validateSegmentBounds(segment));
    }

    print("✓ Large dataset performance test passed\n", .{});
}

// Test case structure for the comprehensive validation
const TestCase = struct {
    name: []const u8,
    data_generator: *const fn (std.mem.Allocator) anyerror![]f64,
    error_bound: f32,
    expected_min_segments: usize,
    expected_max_segments: usize,
};

// Data generator functions for comprehensive test
fn genLinearTrend(alloc: std.mem.Allocator) ![]f64 {
    return generateLinearData(alloc, 0.0, 20.0, 50);
}

fn genSineWave(alloc: std.mem.Allocator) ![]f64 {
    return generateSineWave(alloc, 3.0, 2.0, 100);
}

fn genStepFunction(alloc: std.mem.Allocator) ![]f64 {
    return generateStepData(alloc, 2.0, 60);
}

// Summary test that runs a comprehensive validation
test "computeSegmentsMetadata - comprehensive validation" {
    print("\n=== Comprehensive Validation Test ===\n", .{});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Test data from the Sim-Piece paper examples
    const test_cases = [_]TestCase{
        TestCase{
            .name = "Linear trend",
            .data_generator = genLinearTrend,
            .error_bound = 0.5,
            .expected_min_segments = 1,
            .expected_max_segments = 10,
        },
        TestCase{
            .name = "Sine wave",
            .data_generator = genSineWave,
            .error_bound = 0.3,
            .expected_min_segments = 10,
            .expected_max_segments = 50,
        },
        TestCase{
            .name = "Step function",
            .data_generator = genStepFunction,
            .error_bound = 0.1,
            .expected_min_segments = 5,
            .expected_max_segments = 30,
        },
    };

    for (test_cases) |test_case| {
        print("\n--- Testing: {s} ---\n", .{test_case.name});

        const data = try test_case.data_generator(allocator);
        defer allocator.free(data);

        var segments_metadata = ArrayList(mix_piece.SegmentMetadata).init(allocator);
        defer segments_metadata.deinit();

        try mix_piece.computeSegmentsMetadata(data, &segments_metadata, test_case.error_bound);

        print("Generated {} segments (expected {}-{})\n", .{ segments_metadata.items.len, test_case.expected_min_segments, test_case.expected_max_segments });

        // Validate segment count is within expected range
        try testing.expect(segments_metadata.items.len >= test_case.expected_min_segments);
        try testing.expect(segments_metadata.items.len <= test_case.expected_max_segments);

        // Validate all segments
        for (segments_metadata.items, 0..) |segment, i| {
            try testing.expect(validateSegmentBounds(segment));

            // Validate that segment start times are in order
            if (i > 0) {
                try testing.expect(segment.start_time > segments_metadata.items[i - 1].start_time);
            }

            // Validate that interception values are quantized properly
            const quantized_floor = @floor(segment.interception / test_case.error_bound) * test_case.error_bound;
            const quantized_ceil = @ceil(segment.interception / test_case.error_bound) * test_case.error_bound;
            const diff_floor = @abs(segment.interception - quantized_floor);
            const diff_ceil = @abs(segment.interception - quantized_ceil);

            // The interception should be close to either floor or ceil quantization
            try testing.expect(diff_floor < 0.001 or diff_ceil < 0.001);
        }

        print("✓ {s} validation passed\n", .{test_case.name});
    }

    print("\n✓ Comprehensive validation test passed\n", .{});
}
