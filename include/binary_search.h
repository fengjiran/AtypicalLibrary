//
// Created by richard on 7/31/24.
//

#ifndef ATYPICALLIBRARY_BINARY_SEARCH_H
#define ATYPICALLIBRARY_BINARY_SEARCH_H

#include <vector>

template<typename T>
class BinarySearch {
public:
    /**
     * @brief Find the index of the first element that greater equal the target.
     * @param nums Ascending array.
     * @param left Left bound of search interval.
     * @param right Right bound of search interval.
     * @param target Target.
     * @return Index.
     */
    static int FindFirstGEItem(const std::vector<T>& nums, int left, int right, T target) {
        int l = left;
        int r = right + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        if (l == right + 1) {
            return -1;
        }

        return l;
    }

    /**
     * @brief Find the index of the first element that greater than the target.
     * @param nums Ascending array.
     * @param left Left bound of search interval.
     * @param right Right bound of search interval.
     * @param target Target.
     * @return Index.
     */
    static int FindFirstGTItem(const std::vector<T>& nums, int left, int right, T target) {
        int l = left;
        int r = right + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] <= target) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        if (l == right + 1) {
            return -1;
        }

        return l;
    }

    static int LeftBound(const std::vector<T>& nums, int left, int right, T target) {
        int l = left;
        int r = right + 1;
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (nums[mid] < target) {
                l = mid + 1;
            } else if (nums[mid] > target) {
                r = mid;
            } else if (nums[mid] == target) {
                r = mid;
            }
        }

        if (l == right + 1) {
            return -1;
        }

        return nums[l] == target ? l : -1;
    }
};

#endif//ATYPICALLIBRARY_BINARY_SEARCH_H
