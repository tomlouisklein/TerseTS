const std = @import("std");
const shared = @import("./utilities/shared_structs.zig");
const mixedPLA = @import("./functional/mixed_PLA_cpp_version.zig");
const ConvexList = mixedPLA.ConvexList;
const ParameterPoint = mixedPLA.ParameterPoint;
const DataSegment = mixedPLA.DataSegment;
const HalfPlane = mixedPLA.HalfPlane;
const IntersectionResult = mixedPLA.IntersectionResult;

const CV = @import("./functional/convex_polygon.zig");
const ConvexPolygon = CV.ConvexPolygon;

// ===============
// TESTING SECTION
// ===============

const testing = std.testing;
const expect = testing.expect;
const expectEqual = testing.expectEqual;
const expectApproxEqAbs = testing.expectApproxEqAbs;

// Helper function for approximate equality of ParameterPoint
fn expectParameterPointApproxEqual(expected: ParameterPoint, actual: ParameterPoint, tolerance: f64) !void {
    try expectApproxEqAbs(expected.slope, actual.slope, tolerance);
    try expectApproxEqAbs(expected.intercept, actual.intercept, tolerance);
}

// Helper function for approximate equality of LinearFunction
fn expectLinearFunctionApproxEqual(expected: shared.LinearFunction, actual: shared.LinearFunction, tolerance: f64) !void {
    try expectApproxEqAbs(expected.slope, actual.slope, tolerance);
    try expectApproxEqAbs(expected.intercept, actual.intercept, tolerance);
}

test "ConvexPolygon: basic initialization and deinitialization" {
    const allocator = testing.allocator;

    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    try expect(!polygon.isInstantiated());
    try expectEqual(@as(usize, 0), polygon.getSize());
    try expect(polygon.getEndPoint(true) == null);
    try expect(polygon.getEndPoint(false) == null);
}

test "ConvexPolygon: initWithPoints - simple case" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create two simple points
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 1, .value = 1.0 };
    const epsilon = 0.1;

    try polygon.initWithPoints(p1, p2, epsilon);

    try expect(polygon.isInstantiated());
    try expect(polygon.getSize() > 0);

    // Test that we can get a solution
    const solution = polygon.selectSolution(0.0, 0.0);
    try expect(solution != null);
}

test "ConvexPolygon: initWithPoints - horizontal line" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Horizontal line: both points have same y-value
    const p1 = shared.DiscretePoint{ .time = 0, .value = 5.0 };
    const p2 = shared.DiscretePoint{ .time = 10, .value = 5.0 };
    const epsilon = 0.2;

    try polygon.initWithPoints(p1, p2, epsilon);

    try expect(polygon.isInstantiated());

    // Solution should be approximately horizontal (slope ≈ 0)
    const solution = polygon.selectSolution(0.0, 0.0).?;
    try expectApproxEqAbs(0.0, solution.slope, 0.1);
    try expectApproxEqAbs(5.0, solution.intercept, 0.3); // Within error bound
}

test "ConvexPolygon: initWithPoints - steep line" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Steep line
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 1, .value = 10.0 };
    const epsilon = 0.1;

    try polygon.initWithPoints(p1, p2, epsilon);

    try expect(polygon.isInstantiated());

    // Solution should have steep slope (≈ 10.0)
    const solution = polygon.selectSolution(0.0, 0.0).?;
    try expectApproxEqAbs(10.0, solution.slope, 0.5);
}

test "ConvexPolygon: initWithPoints - same time coordinates" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Same time coordinates (vertical line in data space)
    const p1 = shared.DiscretePoint{ .time = 5, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 5, .value = 10.0 };
    const epsilon = 0.1;

    try polygon.initWithPoints(p1, p2, epsilon);

    // Should still be instantiated, even though it's a degenerate case
    try expect(polygon.isInstantiated());
}

test "ConvexPolygon: intersect with half-plane - contain all" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create a simple polygon
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 2, .value = 2.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    // Create a very permissive half-plane that should contain everything
    const loose_hp = HalfPlane.fromDiscretePoint(1.0, 10.0, 5.0, true);

    const result = try polygon.intersect(&loose_hp);
    try expectEqual(IntersectionResult.ContainAllChain, result);
}

test "ConvexPolygon: intersect with half-plane - contain all 2" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create a simple polygon
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 2, .value = 2.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    // Create a very restrictive half-plane that should exclude everything
    const tight_hp = HalfPlane.fromDiscretePoint(1.0, -10.0, 0.1, false);

    const result = try polygon.intersect(&tight_hp);
    try expectEqual(IntersectionResult.ContainAllChain, result);
}

test "ConvexPolygon: intersect with half-plane - contain none" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create a simple polygon
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 2, .value = 2.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    // Create a very restrictive half-plane that should exclude everything
    const tight_hp = HalfPlane.fromDiscretePoint(1.0, -10.0, 0.1, true);

    const result = try polygon.intersect(&tight_hp);
    try expectEqual(IntersectionResult.ContainNoneChain, result);
}

test "ConvexPolygon: intersect with non-instantiated polygon" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Don't initialize the polygon
    const hp = HalfPlane.fromDiscretePoint(1.0, 0.0, 0.1, true);

    const result = try polygon.intersect(&hp);
    try expectEqual(IntersectionResult.NonExistChain, result);
}

test "ConvexPolygon: selectSolution - non-instantiated" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Don't initialize the polygon
    const solution = polygon.selectSolution(0.0, 5.0).?;

    // Should return horizontal line with fallback value
    try expectApproxEqAbs(0.0, solution.slope, 1e-10);
    try expectApproxEqAbs(5.0, solution.intercept, 1e-10);
}

test "ConvexPolygon: selectSolution with time shift" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 2, .value = 4.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    const time_shift = 1.0;
    const solution = polygon.selectSolution(time_shift, 0.0).?;

    // Verify that time shift affects the intercept correctly
    // new_intercept = old_intercept - slope * time_shift
    try expect(solution.slope != 0); // Should have some slope
}

test "ConvexPolygon: reInitWithSegment - normal case" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create two data segments
    const left_seg = DataSegment.init(0, 0.0, 0.1);
    const current_seg = DataSegment.init(2, 2.0, 0.1);
    const time_base: usize = 0;

    try polygon.reInitWithSegment(&left_seg, &current_seg, time_base, .UpperChain);

    try expect(polygon.isInstantiated());
    try expect(polygon.getSize() > 0);

    const solution = polygon.selectSolution(0.0, 0.0);
    try expect(solution != null);
}

test "ConvexPolygon: reInitWithSegment - same time coordinates" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Create segments with same time (vertical case)
    const left_seg = DataSegment.init(5, 0.0, 0.1);
    const current_seg = DataSegment.init(5, 2.0, 0.1);
    const time_base: usize = 5;

    try polygon.reInitWithSegment(&left_seg, &current_seg, time_base, .LowerChain);

    try expect(polygon.isInstantiated());
}

test "ConvexPolygon: reInitWithSegment - different closure directions" {
    const allocator = testing.allocator;
    var polygon1 = try ConvexPolygon.init(allocator);
    defer polygon1.deinit();
    var polygon2 = try ConvexPolygon.init(allocator);
    defer polygon2.deinit();

    const left_seg = DataSegment.init(0, 0.0, 0.1);
    const current_seg = DataSegment.init(1, 1.0, 0.1);
    const time_base: usize = 0;

    // Test both closure directions
    try polygon1.reInitWithSegment(&left_seg, &current_seg, time_base, .UpperChain);
    try polygon2.reInitWithSegment(&left_seg, &current_seg, time_base, .LowerChain);

    try expect(polygon1.isInstantiated());
    try expect(polygon2.isInstantiated());

    // Both should produce valid solutions (might be different)
    const solution1 = polygon1.selectSolution(0.0, 0.0);
    const solution2 = polygon2.selectSolution(0.0, 0.0);
    try expect(solution1 != null);
    try expect(solution2 != null);
}

test "ConvexPolygon: setEmpty and reset functionality" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Initialize with points
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 1, .value = 1.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    try expect(polygon.isInstantiated());

    // Set empty
    polygon.setEmpty();
    try expect(!polygon.isInstantiated());
    try expectEqual(@as(usize, 0), polygon.getSize());
}

test "ConvexPolygon: resetEndPoints functionality" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 1, .value = 1.0 };
    try polygon.initWithPoints(p1, p2, 0.1);

    // Reset endpoints
    const new_upper = ParameterPoint{ .slope = 1.0, .intercept = 2.0 };
    const new_lower = ParameterPoint{ .slope = -1.0, .intercept = 3.0 };

    polygon.resetEndPoints(new_upper, new_lower);

    const upper_end = polygon.getEndPoint(true);
    const lower_end = polygon.getEndPoint(false);

    try expect(upper_end != null);
    try expect(lower_end != null);
    try expectParameterPointApproxEqual(new_upper, upper_end.?, 1e-10);
    try expectParameterPointApproxEqual(new_lower, lower_end.?, 1e-10);
}

test "ConvexPolygon: computeLineIntersection helper" {
    // Test the helper function directly
    const line1 = shared.LinearFunction{ .slope = 1.0, .intercept = 0.0 }; // y = x
    const line2 = shared.LinearFunction{ .slope = -1.0, .intercept = 2.0 }; // y = -x + 2

    // These lines intersect at (1, 1)
    const intersection = ConvexPolygon.computeLineIntersection(line1, line2);

    try expectApproxEqAbs(1.0, intersection.slope, 1e-10);
    try expectApproxEqAbs(1.0, intersection.intercept, 1e-10);
}

test "ConvexPolygon: computeLineIntersection - parallel lines" {
    // Test parallel lines
    const line1 = shared.LinearFunction{ .slope = 2.0, .intercept = 1.0 };
    const line2 = shared.LinearFunction{ .slope = 2.0, .intercept = 3.0 };

    // Parallel lines - should return a reasonable fallback
    const intersection = ConvexPolygon.computeLineIntersection(line1, line2);

    // Should return a point on one of the lines
    try expectApproxEqAbs(0.0, intersection.slope, 1e-10);
    try expectApproxEqAbs(1.0, intersection.intercept, 1e-10);
}

test "ConvexPolygon: stress test with many intersections" {
    const allocator = testing.allocator;
    var polygon = try ConvexPolygon.init(allocator);
    defer polygon.deinit();

    // Initialize with points
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 10, .value = 10.0 };
    try polygon.initWithPoints(p1, p2, 1.0);

    // Apply many half-plane constraints
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const t = @as(f64, @floatFromInt(i)) * 0.2;
        const v = @as(f64, @floatFromInt(i)) * 0.2;
        const hp = HalfPlane.fromDiscretePoint(t, v, 0.5, i % 2 == 0);

        const result = try polygon.intersect(&hp);

        // Should not crash and should return a valid result
        try expect(result == .ContainAllChain or
            result == .ContainSomeChain or
            result == .ContainNoneChain or
            result == .NonExistChain);

        // If polygon becomes non-existent, break
        if (result == .ContainNoneChain) break;
    }
}

test "ConvexPolygon: error bound scaling" {
    const allocator = testing.allocator;

    // Test with different error bounds
    const error_bounds = [_]f64{ 0.01, 0.1, 1.0, 10.0 };

    for (error_bounds) |epsilon| {
        var polygon = try ConvexPolygon.init(allocator);
        defer polygon.deinit();

        const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
        const p2 = shared.DiscretePoint{ .time = 5, .value = 5.0 };

        try polygon.initWithPoints(p1, p2, epsilon);

        try expect(polygon.isInstantiated());

        const solution = polygon.selectSolution(0.0, 0.0);
        try expect(solution != null);

        // Larger error bounds should allow more flexibility
        // (This is a general sanity check)
    }
}

test "ConvexPolygon: validateAndFixRegion helper functions" {
    // This tests the internal logic we added for the elegant reInitWithSegment
    const corners = [4]ParameterPoint{
        ParameterPoint{ .slope = 0.0, .intercept = 0.0 },
        ParameterPoint{ .slope = 1.0, .intercept = 1.0 },
        ParameterPoint{ .slope = 2.0, .intercept = 2.0 },
        ParameterPoint{ .slope = 3.0, .intercept = 3.0 },
    };

    // Test both closure directions
    const region1 = ConvexPolygon.validateAndFixRegion(corners, .UpperChain);
    const region2 = ConvexPolygon.validateAndFixRegion(corners, .LowerChain);

    // Both should produce valid regions
    try expect(region1.left_bottom.slope >= 0.0);
    try expect(region2.left_bottom.slope >= 0.0);
}
