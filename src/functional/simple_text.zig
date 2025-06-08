const std = @import("std");

pub fn main() !void {
    std.debug.print("Hello, World!\n", .{});

    // Simple test - no imports yet
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    std.debug.print("Test data: ", .{});
    for (data) |val| {
        std.debug.print("{d} ", .{val});
    }
    std.debug.print("\n", .{});
}
