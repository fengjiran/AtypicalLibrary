#include <gtest/gtest.h>

class KMP {
public:
    explicit KMP(const std::string& pat) : n(pat.size()) {
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
    }

    [[nodiscard]] int search(const std::string& txt) const {
        auto m = txt.size();
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            j = dp[j][txt[i]];
            ++i;
        }
        return j == n ? static_cast<int>(i - n) : -1;
    }

private:
    std::vector<std::vector<int>> dp;
    size_t n;
};
