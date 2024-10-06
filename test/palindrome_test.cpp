#include <gtest/gtest.h>

class Palindrome {
public:
    enum class Method {
        kManacher,         // 使用马拉车算法
        kDoublePointer,    // 使用双指针算法
        kDynamicProgramming// 使用动态规划算法
    };

    explicit Palindrome(const std::string& str) : s(str), len(static_cast<int>(s.size())) {}

    Palindrome() = delete;
    Palindrome(const Palindrome&) = delete;
    Palindrome(Palindrome&&) = delete;
    Palindrome& operator=(const Palindrome&) = delete;
    Palindrome& operator=(Palindrome&&) = delete;

    bool IsPalindromicString(Method m = Method::kDoublePointer) {
        return IsPalindromicString(0, len - 1, m);
    }

    bool IsPalindromicString(int left, int right, Method m = Method::kDoublePointer) {
        bool res = true;
        switch (m) {
            case Method::kDoublePointer: {
                while (left < right) {
                    if (s[left] != s[right]) {
                        res = false;
                        break;
                    }
                    ++left;
                    --right;
                }
                break;
            }

            case Method::kDynamicProgramming: {
                GenerateDPTable();
                res = dp[left][right];
                break;
            }

            default:
                break;
        }

        return res;
    }

    int PalindromicSubstringNum(Method m = Method::kManacher) {
        int nums = 0;

        switch (m) {
            case Method::kManacher: {
                std::string t = "$#";
                for (char c: s) {
                    t += c;
                    t += '#';
                }
                t += '!';

                int n = static_cast<int>(t.size());
                std::vector<int> radius(n);
                int center = 0;
                int right = 0;

                for (int i = 1; i < n - 1; ++i) {
                    radius[i] = i <= right ? std::min(radius[2 * center - i], right - i + 1) : 1;
                    while (t[i + radius[i]] == t[i - radius[i]]) {
                        ++radius[i];
                    }

                    if (i + radius[i] - 1 > right) {
                        center = i;
                        right = i + radius[i] - 1;
                    }

                    nums += radius[i] / 2;
                }

                break;
            }

            case Method::kDoublePointer: {
                for (int i = 0; i < 2 * len - 1; ++i) {
                    int left = i / 2;
                    int right = left + i % 2;
                    while (left >= 0 && right < len && s[left] == s[right]) {
                        --left;
                        ++right;
                    }
                    nums += (right - left) / 2;
                }
                break;
            }

            case Method::kDynamicProgramming: {
                if (!dpTableIsGenerated) {
                    dp.resize(len, std::vector<bool>(len));
                    for (int i = len - 1; i >= 0; --i) {
                        for (int j = i; j < len; ++j) {
                            dp[i][j] = s[i] == s[j] && (j - i < 3 || dp[i + 1][j - 1]);
                            if (dp[i][j]) {
                                ++nums;
                            }
                        }
                    }
                    dpTableIsGenerated = true;
                } else {
                    for (int i = len - 1; i >= 0; --i) {
                        for (int j = i; j < len; ++j) {
                            if (dp[i][j]) {
                                ++nums;
                            }
                        }
                    }
                }

                break;
            }

            default:
                break;
        }

        return nums;
    }

private:
    void GenerateDPTable() {
        if (!dpTableIsGenerated) {
            dp.resize(len, std::vector<bool>(len));
            for (int i = len - 1; i >= 0; --i) {
                for (int j = i; j < len; ++j) {
                    dp[i][j] = s[i] == s[j] && (j - i < 3 || dp[i + 1][j - 1]);
                }
            }
            dpTableIsGenerated = true;
        }
    }

    const std::string& s;
    int len;
    std::vector<std::vector<bool>> dp;
    bool dpTableIsGenerated{false};
};

TEST(PalindromeTest, test1) {
    std::string s = "aba";
    Palindrome p(s);
    EXPECT_TRUE(p.IsPalindromicString(Palindrome::Method::kDynamicProgramming));
}

TEST(PalindromeTest, test2) {
    std::string s1 = "abc";
    Palindrome p1(s1);
    EXPECT_EQ(p1.PalindromicSubstringNum(Palindrome::Method::kDynamicProgramming), 3);

    std::string s2 = "aaa";
    Palindrome p2(s2);
    EXPECT_EQ(p2.PalindromicSubstringNum(Palindrome::Method::kDynamicProgramming), 6);
}
