#ifndef AUXFUNCTIONS_H
#define AUXFUNCTIONS_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

std::vector<double> knn(const std::vector<double>& points, int k);
std::vector<bool> getPointsMask(const std::vector<double>& points, double threshold, int k);
std::pair<int, int> getBestLine(const std::vector<std::vector<double>>& linePoints, double threshold, int k, bool rec);
#endif /* AUXFUNCTIONS_H */