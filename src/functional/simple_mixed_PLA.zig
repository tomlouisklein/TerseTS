// Copyright 2025 TerseTS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Implementation of the Mix-Piece algorithm from the paper
//! “G. Luo, K. Yi, S.-W. Cheng, Z. Li, W. Fan, C. He, and Y. Mu.
//! Piecewise Linear Approximation of Streaming Time Series Data with Max‑Error Guarantees.
//! Proc. IEEE 31st Int’l Conf. Data Engineering (ICDE)*, Seoul, South Korea 2015, pp. 173–184.
//! https://ieeexplore.ieee.org/document/7113282”.
//!
//! The implementation is partially based on the author's implementation generously provided
//! by Prof. Ke Yi of Hong Kong University of Science and Technology.const std = @import("std");

const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const mem = std.mem;
const Error = tersets.Error;

const tersets = @import("../tersets.zig");
const shared = @import("../utilities/shared_structs.zig");
const tester = @import("../tester.zig");
const ch = @import("../utilities/convex_hull.zig");

// compress() function

// decompress() function

// ====================
// CORE DATA STRUCTURES
// ====================

// Main convex polygon structure
pub const ConvexPolygon = struct {
    upper_edges: ConvexList,
    lower_edges: ConvexList,
    instantiated: bool,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !ConvexPolygon {
        return .{
            .upper_edges = try ConvexList.init(allocator, true),
            .lower_edges = try ConvexList.init(allocator, false),
            .instantiated = false,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConvexPolygon) void {
        self.upper_edges.deinit();
        self.lower_edges.deinit();
    }
};

// Represents a point with upper and lower bounds (±epsilon).
pub const DataSegment = struct {
    upper: shared.DiscretePoint, // (time, value + epsilon).
    lower: shared.DiscretePoint, // (time, value - epsilon).

    pub fn init(time: usize, value: f64, epsilon: f64) DataSegment {
        return .{
            .upper = shared.DiscretePoint{ .time = time, .value = value + epsilon },
            .lower = shared.DiscretePoint{ .time = time, .value = value - epsilon },
        };
    }

    pub fn getValue(self: *const DataSegment) f64 {
        return (self.upper.value + self.lower.value) / 2.0;
    }
};

// Point in parameter space (slope, intercept).
pub const ParameterPoint = struct {
    slope: f64,
    intercept: f64,

    // Convert to/from LinearFunction for clarity.
    pub fn fromLinearFunction(lf: shared.LinearFunction) ParameterPoint {
        return .{ .slope = lf.slope, .intercept = lf.intercept };
    }

    pub fn toLinearFunction(self: ParameterPoint) shared.LinearFunction {
        return .{ .slope = self.slope, .intercept = self.intercept };
    }
};

// const ConvexPolygon

// const VisibleRegion

// visibleRegion.update()

// const Window

// nextWindow

// closingWindow

// const halfPlane

// ======================
// CORE ALGORITHM DETAILS
// ======================
