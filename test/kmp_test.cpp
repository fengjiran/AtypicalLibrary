#include <gtest/gtest.h>

class KMP {
public:
    explicit KMP(const std::string& pat) : pat_(pat) {
        auto n = pat_.size();
        dp.resize(n, std::vector<int>(256));
        dp[0][pat[0]] = 1;

        int x = 0;
        for (int j = 1; j < n; ++j) {
            for (int c = 0; c < 256; ++c) {
                dp[j][c] = dp[x][c];
            }
            dp[j][pat[j]] = j + 1;
            x = dp[x][pat[j]];
        }

        next.resize(n);
        int i = 1;
        int j = 0;
        while (i < n) {
            if (pat[i] == pat[j]) {
                next[i] = j + 1;
                ++i;
                ++j;
            } else {
                if (j == 0) {
                    next[i] = 0;
                    ++i;
                } else {
                    j = next[j - 1];
                }
            }
        }
    }

    [[nodiscard]] int search(const std::string& txt) const {
        auto m = txt.size();
        auto n = pat_.size();
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            j = dp[j][txt[i]];
            ++i;
        }
        return j == n ? static_cast<int>(i - n) : -1;
    }

    [[nodiscard]] int search2(const std::string& txt) const {
        auto m = txt.size();
        auto n = pat_.size();
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            if (txt[i] == pat_[j]) {
                ++i;
                ++j;
            } else {
                if (j == 0) {
                    ++i;
                } else {
                    j = next[j - 1];
                }
            }
        }
        return j == n ? static_cast<int>(i - n) : -1;
    }

private:
    std::vector<std::vector<int>> dp;
    std::vector<int> next;
    const std::string& pat_;
};

