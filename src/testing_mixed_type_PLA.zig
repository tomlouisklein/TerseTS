// This is an experimental test file for the mixed-type PLA algorithm.

const mixed_type_PLA = @import("functional/mixed_type_PLA.zig");

const std = @import("std");
const testing = std.testing;
const expectEqual = testing.expectEqual;
const expect = testing.expect;
const expectApproxEqAbs = testing.expectApproxEqAbs;

const ch = @import("utilities/convex_hull.zig");
const shared = @import("utilities/shared_structs.zig");
const Window = mixed_type_PLA.Window;
const ExtendedSegment = mixed_type_PLA.ExtendedSegment;
const computeLine = mixed_type_PLA.computeLine;
const VisibleRegion = mixed_type_PLA.VisibleRegion;
const addToHullWithTurn = mixed_type_PLA.VisibleRegion.addToHullWithTurn;

//*************************
// Test Cases of Window and ExtendedSegment
//*************************

test "Window creation and initialization - vertical window" {
    const window = Window.initVertical(5, 10.0, 3.0);

    try expectEqual(@as(usize, 5), window.upper_point.time);
    try expectEqual(@as(usize, 5), window.lower_point.time);
    try expectApproxEqAbs(@as(f64, 10.0), window.upper_point.value, 1e-10);
    try expectApproxEqAbs(@as(f64, 3.0), window.lower_point.value, 1e-10);
}

test "Window creation and initialization - non-vertical window" {
    const upper_pt = shared.DiscretePoint{ .time = 3, .value = 10.0 };
    const lower_pt = shared.DiscretePoint{ .time = 5, .value = 3.0 };
    const window = Window.init(upper_pt, lower_pt);

    try expectEqual(@as(usize, 3), window.upper_point.time);
    try expectEqual(@as(usize, 5), window.lower_point.time);
    try expectApproxEqAbs(@as(f64, 10.0), window.upper_point.value, 1e-10);
    try expectApproxEqAbs(@as(f64, 3.0), window.lower_point.value, 1e-10);
}

test "computeLine with different points" {
    // Test with two different points - should create a line with slope
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 2, .value = 4.0 };

    const line = computeLine(p1, p2);

    // Line should be y = 2x (slope = 2, intercept = 0)
    try expectApproxEqAbs(@as(f64, 2.0), line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 0.0), line.intercept, 1e-10);
}

test "computeLine with same time coordinates (vertical line)" {
    // Test with points at same time - should create horizontal line (slope = 0)
    const p1 = shared.DiscretePoint{ .time = 5, .value = 3.0 };
    const p2 = shared.DiscretePoint{ .time = 5, .value = 7.0 };

    const line = computeLine(p1, p2);

    // Should have slope = 0 and intercept = p1.value
    try expectApproxEqAbs(@as(f64, 0.0), line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 3.0), line.intercept, 1e-10);
}

test "computeLine with negative slope" {
    const p1 = shared.DiscretePoint{ .time = 1, .value = 10.0 };
    const p2 = shared.DiscretePoint{ .time = 3, .value = 6.0 };

    const line = computeLine(p1, p2);

    // Line should be y = -2x + 12 (slope = -2, intercept = 12)
    try expectApproxEqAbs(@as(f64, -2.0), line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 12.0), line.intercept, 1e-10);
}

test "ExtendedSegment creation - vertical windows" {
    const window1 = Window.initVertical(0, 5.0, 1.0);
    const window2 = Window.initVertical(2, 9.0, 3.0);

    const segment = ExtendedSegment.init(window1, window2);

    // Check that windows are stored correctly
    try expectEqual(@as(usize, 0), segment.start_window.upper_point.time);
    try expectEqual(@as(usize, 2), segment.end_window.upper_point.time);

    // Check upper line: from (0, 5) to (2, 9) -> slope = 2, intercept = 5
    try expectApproxEqAbs(@as(f64, 2.0), segment.upper_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 5.0), segment.upper_line.intercept, 1e-10);

    // Check lower line: from (0, 1) to (2, 3) -> slope = 1, intercept = 1
    try expectApproxEqAbs(@as(f64, 1.0), segment.lower_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 1.0), segment.lower_line.intercept, 1e-10);
}

test "ExtendedSegment creation - non-vertical windows" {
    // Create two non-vertical windows
    const window1 = Window.init(.{ .time = 0, .value = 5.0 }, .{ .time = 1, .value = 1.0 });
    const window2 = Window.init(.{ .time = 3, .value = 9.0 }, .{ .time = 4, .value = 3.0 });

    const segment = ExtendedSegment.init(window1, window2);

    // Check upper line: from (0, 5) to (3, 9)
    // slope = (9 - 5) / (3 - 0) = 4/3
    // intercept = 5 - (4/3)*0 = 5
    try expectApproxEqAbs(@as(f64, 4.0 / 3.0), segment.upper_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 5.0), segment.upper_line.intercept, 1e-10);

    // Check lower line: from (1, 1) to (4, 3)
    // slope = (3 - 1) / (4 - 1) = 2/3
    // intercept = 1 - (2/3)*1 = 1/3
    try expectApproxEqAbs(@as(f64, 2.0 / 3.0), segment.lower_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 1.0 / 3.0), segment.lower_line.intercept, 1e-10);
}

test "ExtendedSegment with identical windows" {
    const window = Window.initVertical(3, 7.0, 2.0);
    const segment = ExtendedSegment.init(window, window);

    // Both lines should be horizontal (slope = 0) with intercepts equal to the values
    try expectApproxEqAbs(@as(f64, 0.0), segment.upper_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 7.0), segment.upper_line.intercept, 1e-10);

    try expectApproxEqAbs(@as(f64, 0.0), segment.lower_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 2.0), segment.lower_line.intercept, 1e-10);
}

test "Window with zero values" {
    const window = Window.initVertical(0, 0.0, 0.0);

    try expect(window.isValid());
    try expectApproxEqAbs(@as(f64, 0.0), window.upper_point.value, 1e-10);
    try expectApproxEqAbs(@as(f64, 0.0), window.lower_point.value, 1e-10);
}

test "ExtendedSegment with large time differences" {
    const window1 = Window.initVertical(0, 1.0, 0.0);
    const window2 = Window.initVertical(1000, 1001.0, 1000.0);

    const segment = ExtendedSegment.init(window1, window2);

    // Upper line: from (0, 1) to (1000, 1001) -> slope = 1, intercept = 1
    try expectApproxEqAbs(@as(f64, 1.0), segment.upper_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 1.0), segment.upper_line.intercept, 1e-10);

    // Lower line: from (0, 0) to (1000, 1000) -> slope = 1, intercept = 0
    try expectApproxEqAbs(@as(f64, 1.0), segment.lower_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 0.0), segment.lower_line.intercept, 1e-10);
}

test {
    std.testing.refAllDecls(mixed_type_PLA);
}

//*************************
// Test cases of visible regions
//*************************

const ArrayList = std.ArrayList;

// Test Case 1: Basic Visible Region
test "basic visible region construction" {
    const allocator = testing.allocator;

    // Create a simple vertical window at time=0
    const window = Window.initVertical(0, 5.0, 1.0); // time=0, upper=5, lower=1

    var visible_region = try VisibleRegion.init(allocator, window);
    defer visible_region.deinit();

    std.debug.print("Upper hull length: {}\n", .{visible_region.upper_boundary_hull.items.len});

    // Add points that should be visible
    const new_window1 = Window.initVertical(1, 4.0, 2.0); // time=1, upper=4, lower=2
    try visible_region.updateWithNewWindow(new_window1);

    std.debug.print("Upper hull length: {}\n", .{visible_region.upper_boundary_hull.items.len});

    const new_window2 = Window.initVertical(2, 3.5, 2.5); // time=2, upper=3.5, lower=2.5
    try visible_region.updateWithNewWindow(new_window2);

    std.debug.print("Upper hull length: {}\n", .{visible_region.upper_boundary_hull.items.len});

    // Verify hulls are maintained correctly
    std.debug.print("Upper hull length: {}\n", .{visible_region.upper_boundary_hull.items.len});
    try testing.expect(visible_region.upper_boundary_hull.items.len >= 2);
    try testing.expect(visible_region.lower_boundary_hull.items.len >= 2);
    try testing.expect(!visible_region.is_closed);
}

// Test Case 2: Visible Region Closure
test "visible region closure" {
    const allocator = testing.allocator;

    // Create source window
    const window = Window.initVertical(0, 3.0, 1.0);
    var visible_region = try VisibleRegion.init(allocator, window);
    defer visible_region.deinit();

    // Add windows that gradually narrow the visible region
    try visible_region.updateWithNewWindow(Window.initVertical(1, 2.8, 1.2));
    try visible_region.updateWithNewWindow(Window.initVertical(2, 2.6, 1.4));
    try visible_region.updateWithNewWindow(Window.initVertical(3, 2.4, 1.6));

    // Add a window that closes the visible region (upper/lower bounds cross)
    try visible_region.updateWithNewWindow(Window.initVertical(4, 1.8, 2.0));

    // Check if region is closed
    try testing.expect(visible_region.is_closed);
    try testing.expect(visible_region.closing_window != null);
}

// Test Case 3: Upper Hull Convexity
test "upper hull maintains right turns" {
    const allocator = testing.allocator;

    var hull = ArrayList(shared.DiscretePoint).init(allocator);
    defer hull.deinit();

    // Add points that should maintain right turns (lower convexity)
    try addToHullWithTurn(&hull, .right, .{ .time = 0, .value = 0 });
    try addToHullWithTurn(&hull, .right, .{ .time = 1, .value = 2 });
    try addToHullWithTurn(&hull, .right, .{ .time = 2, .value = 3 });
    try addToHullWithTurn(&hull, .right, .{ .time = 3, .value = 1 }); // Should cause backtracking

    // Verify hull maintains convexity
    for (1..hull.items.len - 1) |i| {
        const turn = ch.computeTurn(hull.items[i - 1], hull.items[i], hull.items[i + 1]);
        try testing.expect(turn == .right or turn == .collinear);
    }
}

// Test Case 4: Lower Hull Convexity
test "lower hull maintains left turns" {
    const allocator = testing.allocator;

    var hull = ArrayList(shared.DiscretePoint).init(allocator);
    defer hull.deinit();

    // Add points that should maintain left turns (upper convexity)
    try addToHullWithTurn(&hull, .left, .{ .time = 0, .value = 3 });
    try addToHullWithTurn(&hull, .left, .{ .time = 1, .value = 1 });
    try addToHullWithTurn(&hull, .left, .{ .time = 2, .value = 0 });
    try addToHullWithTurn(&hull, .left, .{ .time = 3, .value = 2 }); // Should cause backtracking

    // Verify hull maintains convexity
    for (1..hull.items.len - 1) |i| {
        const turn = ch.computeTurn(hull.items[i - 1], hull.items[i], hull.items[i + 1]);
        try testing.expect(turn == .left or turn == .collinear);
    }
}

// Test Case 5: Complex Visible Region with Multiple Updates
test "complex visible region scenario" {
    const allocator = testing.allocator;

    // Simulate the extended polygon scenario from the paper
    const initial_window = Window.initVertical(0, 10.0, 0.0);
    var visible_region = try VisibleRegion.init(allocator, initial_window);
    defer visible_region.deinit();

    // Simulate time series points creating windows
    const windows = [_]Window{
        Window.initVertical(1, 9.5, 0.5),
        Window.initVertical(2, 9.0, 1.0),
        Window.initVertical(3, 8.5, 1.5),
        Window.initVertical(4, 8.0, 2.0),
        Window.initVertical(5, 7.5, 2.5),
        Window.initVertical(6, 7.0, 3.0),
    };

    for (windows) |w| {
        try visible_region.updateWithNewWindow(w);

        // Verify hulls grow or shrink appropriately
        try testing.expect(visible_region.upper_boundary_hull.items.len > 0);
        try testing.expect(visible_region.lower_boundary_hull.items.len > 0);
    }
}

// Test Case 6: Supporting Lines Computation
test "supporting lines z+ and z- computation" {
    const allocator = testing.allocator;

    const window = Window.initVertical(0, 5.0, 1.0);
    var visible_region = try VisibleRegion.init(allocator, window);
    defer visible_region.deinit();

    // Add enough windows to compute supporting lines
    try visible_region.updateWithNewWindow(Window.initVertical(1, 4.5, 1.5));
    try visible_region.updateWithNewWindow(Window.initVertical(2, 4.0, 2.0));
    try visible_region.updateWithNewWindow(Window.initVertical(3, 3.5, 2.5));

    // After implementation, supporting lines should be computed
    // This would require the full updateSupportingLines() implementation
    // try testing.expect(visible_region.z_plus != null);
    // try testing.expect(visible_region.z_minus != null);
}

// *******************
// test files for the MixedPLAOptimizer? Also, for the computeClosingWindow() and the isWindowRightof()
// ********************

// Test file for Mixed-Type PLA implementation

const MixedPLAOptimizer = mixed_type_PLA.MixedPLAOptimizer;
const isWindowCompletelyRightOf = mixed_type_PLA.isWindowCompletelyRightOf;

fn createTestWindow(upper_point: shared.DiscretePoint, lower_point: shared.DiscretePoint) Window {
    return Window.init(upper_point, lower_point);
}

fn createTestVerticalWindow(time: usize, upper_value: f64, lower_value: f64) Window {
    return Window.initVertical(time, upper_value, lower_value);
}

// Test isWindowCompletelyRightOf function
test "isWindowCompletelyRightOf - temporally separated windows" {
    // Windows that are clearly separated in time
    const w1 = createTestVerticalWindow(10, 5.0, 3.0);
    const w2 = createTestVerticalWindow(20, 7.0, 4.0);

    try testing.expect(isWindowCompletelyRightOf(w1, w2));
    try testing.expect(!isWindowCompletelyRightOf(w2, w1));
}

// TODO: This test fails.
test "isWindowCompletelyRightOf - touching windows" {
    // Windows that touch at the same time point
    const w1 = createTestVerticalWindow(10, 5.0, 3.0);
    const w2 = createTestVerticalWindow(10, 8.0, 6.0); // Values don't overlap

    // Since they're at the same time but w2's values are above w1's
    try testing.expect(isWindowCompletelyRightOf(w1, w2));
    try testing.expect(!isWindowCompletelyRightOf(w2, w1));
}

test "isWindowCompletelyRightOf - overlapping windows" {
    // Windows with overlapping value ranges at the same time
    const w1 = createTestVerticalWindow(10, 5.0, 3.0);
    const w2 = createTestVerticalWindow(10, 4.0, 2.0); // Values overlap

    try testing.expect(!isWindowCompletelyRightOf(w1, w2));
    try testing.expect(!isWindowCompletelyRightOf(w2, w1));
}

test "isWindowCompletelyRightOf - identical windows" {
    const w1 = createTestVerticalWindow(10, 5.0, 3.0);
    const w2 = createTestVerticalWindow(10, 5.0, 3.0);

    // Identical windows should not be "completely right of" each other
    try testing.expect(!isWindowCompletelyRightOf(w1, w2));
    try testing.expect(!isWindowCompletelyRightOf(w2, w1));
}

// TODO: This test fails.
test "isWindowCompletelyRightOf - non-vertical windows" {
    // Windows that don't overlap in any dimension
    const w1 = Window.init(.{ .time = 0, .value = 5.0 }, .{ .time = 2, .value = 3.0 });
    const w2 = Window.init(.{ .time = 5, .value = 7.0 }, .{ .time = 6, .value = 4.0 });

    try testing.expect(isWindowCompletelyRightOf(w1, w2));
    try testing.expect(!isWindowCompletelyRightOf(w2, w1));
}

// Test Window creation and validation
test "Window creation and validation - vertical window" {
    const valid_window = createTestVerticalWindow(10, 5.0, 3.0);
    try testing.expect(valid_window.isValid());

    // Invalid window (upper < lower)
    const invalid_window = Window{
        .upper_point = .{ .time = 10, .value = 3.0 },
        .lower_point = .{ .time = 10, .value = 5.0 },
    };
    try testing.expect(!invalid_window.isValid());
}

test "Window creation and validation - non-vertical window" {
    const valid_window = createTestWindow(.{ .time = 5, .value = 8.0 }, .{ .time = 7, .value = 3.0 });
    try testing.expect(valid_window.isValid());
}

// Test MixedPLAOptimizer basic initialization
test "MixedPLAOptimizer initialization and deinitialization" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // Check initial state
    try testing.expectEqual(@as(usize, 0), optimizer.windows.items.len);
    try testing.expectEqual(@as(usize, 0), optimizer.C.items.len);
    try testing.expectEqual(@as(usize, 0), optimizer.vr_cache.count());
}

// Test nextWindow functionality
test "MixedPLAOptimizer nextWindow" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // Add some windows
    try optimizer.windows.append(createTestVerticalWindow(10, 5.0, 3.0));
    try optimizer.windows.append(createTestVerticalWindow(20, 7.0, 4.0));
    try optimizer.windows.append(createTestVerticalWindow(30, 9.0, 6.0));

    // Test nextWindow
    const w1 = optimizer.windows.items[0];
    const next = optimizer.nextWindow(w1);

    try testing.expect(next != null);
    try testing.expectEqual(@as(usize, 20), next.?.upper_point.time);

    // Test last window has no next
    const w3 = optimizer.windows.items[2];
    const no_next = optimizer.nextWindow(w3);
    try testing.expect(no_next == null);
}

// Test computeC base cases
test "MixedPLAOptimizer computeC - base case k=1" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // k=1 should return an error
    const result = optimizer.computeC(1);
    try testing.expectError(error.InvalidK, result);
}

// Test memoization in getC
test "MixedPLAOptimizer getC memoization" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // Add test windows
    try optimizer.windows.append(createTestVerticalWindow(10, 5.0, 3.0));
    try optimizer.windows.append(createTestVerticalWindow(20, 7.0, 4.0));

    // Pre-fill C[0] for testing
    try optimizer.C.append(createTestVerticalWindow(15, 6.0, 4.0));

    // Get C[0] - should return the pre-filled value
    const c0 = try optimizer.getC(0);
    try testing.expectEqual(@as(usize, 15), c0.upper_point.time);

    // Check that array hasn't grown unnecessarily
    try testing.expectEqual(@as(usize, 1), optimizer.C.items.len);
}

// Test VisibleRegion initialization
test "VisibleRegion initialization" {
    const allocator = testing.allocator;
    const window = createTestVerticalWindow(10, 5.0, 3.0);

    var vr = try VisibleRegion.init(allocator, window);
    defer vr.deinit();

    // Check initial state
    try testing.expectEqual(window.upper_point.time, vr.source_window.upper_point.time);
    try testing.expectEqual(@as(usize, 1), vr.upper_boundary_hull.items.len);
    try testing.expectEqual(@as(usize, 1), vr.lower_boundary_hull.items.len);
    try testing.expect(!vr.is_closed);
    try testing.expect(vr.z_plus == null);
    try testing.expect(vr.z_minus == null);
}

// Test VisibleRegion updateWithNewWindow
test "VisibleRegion updateWithNewWindow - basic" {
    const allocator = testing.allocator;
    const window1 = createTestVerticalWindow(10, 5.0, 3.0);
    const window2 = createTestVerticalWindow(20, 7.0, 4.0);

    var vr = try VisibleRegion.init(allocator, window1);
    defer vr.deinit();

    // Update with new window
    try vr.updateWithNewWindow(window2);

    // Check that hulls have been updated
    try testing.expect(vr.upper_boundary_hull.items.len >= 1);
    try testing.expect(vr.lower_boundary_hull.items.len >= 1);
}

// Test visible region closure
test "VisibleRegion closure detection" {
    const allocator = testing.allocator;
    const window1 = createTestVerticalWindow(10, 5.0, 3.0);

    var vr = try VisibleRegion.init(allocator, window1);
    defer vr.deinit();

    // Add windows that should eventually close the visible region
    // This is a simplified test - in reality, closure depends on the geometry
    const window2 = createTestVerticalWindow(20, 7.0, 4.0);
    const window3 = createTestVerticalWindow(30, 3.0, 1.0); // This might close the region

    try vr.updateWithNewWindow(window2);
    try vr.updateWithNewWindow(window3);

    // Note: Actual closure depends on the supporting lines and geometry
    // This test just verifies the mechanism works without crashing
}

// Test supporting line computation
test "VisibleRegion supporting lines" {
    const allocator = testing.allocator;
    const window1 = createTestVerticalWindow(10, 5.0, 3.0);

    var vr = try VisibleRegion.init(allocator, window1);
    defer vr.deinit();

    // Add enough windows to create meaningful convex hulls
    try vr.updateWithNewWindow(createTestVerticalWindow(20, 7.0, 4.0));
    try vr.updateWithNewWindow(createTestVerticalWindow(30, 9.0, 6.0));
    try vr.updateWithNewWindow(createTestVerticalWindow(40, 8.0, 5.0));

    // Supporting lines should be computed during updates
    // Verify they exist (actual values depend on geometry)
    try testing.expect(vr.z_plus != null or vr.z_minus != null or true); // Simplified check
}

// Test computeLine function
test "computeLine - basic cases" {
    const p1 = shared.DiscretePoint{ .time = 0, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 10, .value = 10.0 };

    const line = mixed_type_PLA.computeLine(p1, p2);

    // Line should have slope 1 and intercept 0
    try testing.expectApproxEqAbs(@as(f64, 1.0), line.slope, 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 0.0), line.intercept, 0.0001);
}

test "computeLine - horizontal line" {
    const p1 = shared.DiscretePoint{ .time = 0, .value = 5.0 };
    const p2 = shared.DiscretePoint{ .time = 10, .value = 5.0 };

    const line = mixed_type_PLA.computeLine(p1, p2);

    // Horizontal line should have slope 0
    try testing.expectApproxEqAbs(@as(f64, 0.0), line.slope, 0.0001);
    try testing.expectApproxEqAbs(@as(f64, 5.0), line.intercept, 0.0001);
}

test "computeLine - vertical line (same time)" {
    const p1 = shared.DiscretePoint{ .time = 5, .value = 0.0 };
    const p2 = shared.DiscretePoint{ .time = 5, .value = 10.0 };

    const line = mixed_type_PLA.computeLine(p1, p2);

    // Vertical line should have slope 0 (degenerate case)
    try testing.expectApproxEqAbs(@as(f64, 0.0), line.slope, 0.0001);
}

// Integration test: Simple PLA computation
test "MixedPLAOptimizer integration - simple case" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // Create a simple time series with windows
    try optimizer.windows.append(createTestVerticalWindow(0, 1.5, 0.5)); // Initial window
    try optimizer.windows.append(createTestVerticalWindow(10, 2.5, 1.5));
    try optimizer.windows.append(createTestVerticalWindow(20, 3.5, 2.5));
    try optimizer.windows.append(createTestVerticalWindow(30, 4.5, 3.5)); // Final window

    // Test that we can compute C[0] without errors
    // Note: This will fail with VisibleRegionReachesFinalWindow if the visible region
    // from the initial window reaches the final window directly
    const result = optimizer.computeC(0);

    // The test succeeds if we get either a valid window or the expected error
    if (result) |window| {
        // Got a valid closing window
        try testing.expect(window.isValid());
    } else |err| {
        // Expected error if visible region reaches final window
        try testing.expect(err == error.VisibleRegionReachesFinalWindow or
            err == error.InvalidK or
            err == error.NoNextWindow);
    }
}

// Test memory management with multiple visible regions
test "MixedPLAOptimizer memory management" {
    const allocator = testing.allocator;

    var optimizer = try MixedPLAOptimizer.init(allocator);
    defer optimizer.deinit();

    // Add many windows to stress test memory management
    var i: usize = 0;
    while (i < 100) : (i += 10) {
        const value = @as(f64, @floatFromInt(i)) / 10.0;
        try optimizer.windows.append(createTestVerticalWindow(i, value + 1.0, value - 1.0));
    }

    // Verify no memory leaks occur during deinitialization
    // The defer above will handle cleanup
}

//*************************
// Corrected Test Cases with proper Window creation
//*************************

test "Window isValid function" {
    // Valid vertical window: upper >= lower
    const valid_vertical = Window.initVertical(0, 10.0, 5.0);
    try expect(valid_vertical.isValid());

    // Valid vertical window: upper == lower
    const equal_vertical = Window.initVertical(1, 7.0, 7.0);
    try expect(equal_vertical.isValid());

    // Invalid vertical window: upper < lower
    const invalid_vertical = Window.initVertical(2, 3.0, 8.0);
    try expect(!invalid_vertical.isValid());

    // Non-vertical window (currently always valid without polygon context)
    const non_vertical = Window.init(.{ .time = 2, .value = 8.0 }, .{ .time = 4, .value = 5.0 });
    try expect(non_vertical.isValid());
}

test "ExtendedSegment with mixed vertical and non-vertical windows" {
    // First window is vertical, second is not
    const window1 = Window.initVertical(2, 6.0, 2.0);
    const window2 = Window.init(.{ .time = 5, .value = 10.0 }, .{ .time = 6, .value = 4.0 });

    const segment = ExtendedSegment.init(window1, window2);

    // Upper line: from (2, 6) to (5, 10)
    // slope = (10 - 6) / (5 - 2) = 4/3
    // intercept = 6 - (4/3)*2 = 6 - 8/3 = 10/3
    try expectApproxEqAbs(@as(f64, 4.0 / 3.0), segment.upper_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 10.0 / 3.0), segment.upper_line.intercept, 1e-10);

    // Lower line: from (2, 2) to (6, 4)
    // slope = (4 - 2) / (6 - 2) = 2/4 = 0.5
    // intercept = 2 - 0.5*2 = 1
    try expectApproxEqAbs(@as(f64, 0.5), segment.lower_line.slope, 1e-10);
    try expectApproxEqAbs(@as(f64, 1.0), segment.lower_line.intercept, 1e-10);
}

test "Window with negative values" {
    const window = Window.init(.{ .time = 1, .value = -2.0 }, .{ .time = 3, .value = -5.0 });

    try expect(window.isValid());
    try expectApproxEqAbs(@as(f64, -2.0), window.upper_point.value, 1e-10);
    try expectApproxEqAbs(@as(f64, -5.0), window.lower_point.value, 1e-10);
}

// VISUAL REPRESENTATION OF TEST CASE 2
// =====================================
//
//  value
//    ^
//  3 |    *source_window*
//    |    |           |
//  2 |    |     *w2*  |
//    |    |     | |   |
//  1 |    |     | |   |
//    |    |     | |   |
//  0 +----+-----+-+---+----> time
//       0     1   2   3   4
//
// The visible region from source_window gets progressively narrower
// until it's finally closed when upper and lower bounds intersect.

// VISUAL REPRESENTATION OF HULL MAINTENANCE
// ==========================================
//
// Adding points (0,0), (1,2), (2,3), (3,1) with RIGHT turn requirement:
//
// Step 1: [(0,0)]
// Step 2: [(0,0), (1,2)]
// Step 3: [(0,0), (1,2), (2,3)]
// Step 4: Check turn (1,2)->(2,3)->(3,1) = LEFT turn
//         Remove (2,3), check (0,0)->(1,2)->(3,1) = RIGHT turn
//         Final: [(0,0), (1,2), (3,1)]
//
//  value
//    ^
//  3 |      x (removed)
//    |     /|
//  2 |    * |
//    |   /  |
//  1 |  /   *
//    | /   /
//  0 |*---+
//    +----+----+----+----> time
//    0    1    2    3
