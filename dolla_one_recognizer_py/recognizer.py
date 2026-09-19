
# The $1 Unistroke Recognizer (Python 3 version)
#
#     Jacob O. Wobbrock, Ph.D.
#     The Information School
#     University of Washington
#     wobbrock@uw.edu
#
#     Andrew D. Wilson, Ph.D.
#     Microsoft Research
#     awilson@microsoft.com
#
#     Yang Li, Ph.D.
#     Department of Computer Science and Engineering
#     University of Washington
#     yangli@cs.washington.edu
#
# The academic publication for the $1 recognizer, and what should be
# used to cite it, is:
#
#     Wobbrock, J.O., Wilson, A.D. and Li, Y. (2007). Gestures without
#     libraries, toolkits or training: A $1 recognizer for user interface
#     prototypes. Proceedings of the ACM Symposium on User Interface
#     Software and Technology (UIST '07). Newport, Rhode Island (October
#     7-10, 2007). New York: ACM Press, pp. 159-168.
#     https://dl.acm.org/citation.cfm?id=1294238
#
# The Protractor enhancement was separately published by Yang Li and programmed
# here by Jacob O. Wobbrock:
#
#     Li, Y. (2010). Protractor: A fast and accurate gesture
#     recognizer. Proceedings of the ACM Conference on Human
#     Factors in Computing Systems (CHI '10). Atlanta, Georgia
#     (April 10-15, 2010). New York: ACM Press, pp. 2169-2172.
#     https://dl.acm.org/citation.cfm?id=1753654
#
# This software is distributed under the "New BSD License" agreement:
#
# Copyright (C) 2007-2012, Jacob O. Wobbrock, Andrew D. Wilson and Yang Li.
# All rights reserved.
#
# Python 3 port Copyright (C) 2026, yuzujelly2222.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the names of the University of Washington nor Microsoft,
#      nor the names of its contributors may be used to endorse or promote
#      products derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Jacob O. Wobbrock OR Andrew D. Wilson
# OR Yang Li OR yuzujelly2222 BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Sequence

import numpy as np

Point = Sequence[float]
Points = Sequence[Point]


class templates_template:
    points: np.ndarray = np.empty((0, 2))
    name: str = ""


class _deg:
    sin = staticmethod(lambda x: np.sin(np.deg2rad(x)))
    cos = staticmethod(lambda x: np.cos(np.deg2rad(x)))
    tan = staticmethod(lambda x: np.tan(np.deg2rad(x)))
    arcsin = staticmethod(lambda x: np.rad2deg(np.arcsin(x)))
    arccos = staticmethod(lambda x: np.rad2deg(np.arccos(x)))
    arctan = staticmethod(lambda x: np.rad2deg(np.arctan(x)))
    arctan2 = staticmethod(lambda y, x: np.rad2deg(np.arctan2(y, x)))


class dolla_one_recognizer:
    def __init__(self, size: float, templates: Sequence[Points], templates_name: Sequence[str], n: int) -> None:
        self.size = size
        self.n = n
        self.templates: list[templates_template] = []
        for points, name in zip(templates, templates_name):
            template = templates_template()
            template.points = self._normalize(points)
            template.name = name
            self.templates.append(template)

    def add_template(self, points: Points, name: str) -> None:
        template = templates_template()
        template.points = self._normalize(points)
        template.name = name
        self.templates.append(template)

    def _get_distance(self, p1: Point, p2: Point) -> float:
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

    def _get_length(self, points: Points) -> float:
        return sum(self._get_distance(points[i-1], points[i]) for i in range(1, len(points)))

    def _get_centroid(self, points: Points) -> list[float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return [sum(xs)/len(points), sum(ys)/len(points)]

    def _get_bounding_box(self, points: Points) -> tuple[float, float]:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_max = max(xs)
        x_min = min(xs)
        y_max = max(ys)
        y_min = min(ys)
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)
        return width, height

    def _all_point_rotate(self, points: Points, theta: float) -> np.ndarray:
        c = self._get_centroid(points)
        new_points = list()
        for point in points:
            qx = (point[0]-c[0])*_deg.cos(theta) - (point[1]-c[1])*_deg.sin(theta) + c[0]
            qy = (point[0]-c[0])*_deg.sin(theta) + (point[1]-c[1])*_deg.cos(theta) + c[1]
            new_points.append([qx,qy])
        return np.array(new_points)

    def _resample(self, points: Points) -> np.ndarray:
        points = [list(p) for p in points]
        ideal = self._get_length(points) / (self.n - 1)
        new_points = [points[0][:]]
        all_distance = 0.0
        i = 1
        while i < len(points):
            distance = self._get_distance(points[i-1], points[i])
            if all_distance + distance >= ideal:
                t = (ideal - all_distance) / distance
                qx = points[i-1][0] + t * (points[i][0] - points[i-1][0])
                qy = points[i-1][1] + t * (points[i][1] - points[i-1][1])
                q = [qx, qy]
                new_points.append(q)
                points.insert(i, q)
                all_distance = 0.0
            else:
                all_distance += distance
            i += 1
        if len(new_points) == self.n - 1:
            new_points.append(points[-1][:])
        return np.array(new_points)

    def _rotate_to_zero(self, points: Points) -> np.ndarray:
        c = self._get_centroid(points)
        theta = _deg.arctan2(c[1]-points[0][1],c[0]-points[0][0])
        return self._all_point_rotate(points,-theta)

    def _scale_to_square(self, points: Points) -> np.ndarray:
        bounding_box = self._get_bounding_box(points)
        new_points = list()
        for point in  points:
            qx = point[0] * (self.size/bounding_box[0])
            qy = point[1] * (self.size/bounding_box[1])
            new_points.append([qx,qy])
        return np.array(new_points)

    def _translate_to_origin(self, points: Points) -> np.ndarray:
        c = self._get_centroid(points)
        new_points = list()
        for point in points:
            qx = point[0] - c[0]
            qy = point[1] - c[1]
            new_points.append([qx,qy])
        return np.array(new_points)

    def _normalize(self, points: Points) -> np.ndarray:
        points = self._resample(points)
        points = self._rotate_to_zero(points)
        points = self._scale_to_square(points)
        points = self._translate_to_origin(points)
        return points

    def recognize(self, points: Points) -> tuple[np.ndarray, str, float]:
        points = self._normalize(points)
        best_point = float('inf')
        for template in self.templates:
            temp_point = self._distance_at_best_angle(points, template.points, -45,45,2)
            if temp_point < best_point:
                best_point = temp_point
                best_template = template.points
                best_template_name = template.name
        score = 1 - best_point / (0.5*((self.size**2+self.size**2)**0.5))
        return best_template, best_template_name, score

    def _distance_at_best_angle(self, points: Points, template: Points, theta_a: float, theta_b: float, theta_d: float) -> float:
        GOLDEN_RATIO = 0.5*(-1+(5)**0.5)
        x_1 = GOLDEN_RATIO*theta_a + (1-GOLDEN_RATIO)*theta_b
        f_1 = self._distance_at_angle(points, template, x_1)
        x_2 = GOLDEN_RATIO*theta_b + (1-GOLDEN_RATIO)*theta_a
        f_2 = self._distance_at_angle(points, template, x_2)
        while abs(theta_b-theta_a) > theta_d:
            if f_1 < f_2:
                theta_b = x_2
                x_2 = x_1
                f_2 = f_1
                x_1 = GOLDEN_RATIO*theta_a +(1-GOLDEN_RATIO)*theta_b
                f_1 = self._distance_at_angle(points, template,x_1)
            else:
                theta_a = x_1
                x_1 = x_2
                f_1 = f_2
                x_2 = GOLDEN_RATIO*theta_b + (1-GOLDEN_RATIO)*theta_a
                f_2 = self._distance_at_angle(points, template, x_2)

        return min(f_1,f_2)

    def _distance_at_angle(self, points: Points, template: Points, theta: float) -> float:
        new_points = self._all_point_rotate(points,theta)
        return self._path_distance(new_points,template)

    def _path_distance(self, a: Points, b: Points) -> float:
        d = 0
        for i in range(len(a)):
            d = d + self._get_distance(a[i],b[i])
        return d / len(a)

    def main(self, points: Points) -> tuple[np.ndarray, str, float]:
        return self.recognize(points)
