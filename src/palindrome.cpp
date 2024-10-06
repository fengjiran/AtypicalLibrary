#include <string>
#include <vector>

class Palindrome {
public:
    enum class Method {
        Manacher,     // 使用马拉车算法
        DoublePointer,// 使用双指针算法
        DP            // 使用动态规划算法
    };

    Palindrome() = delete;
    Palindrome(const std::string& str) : s(str) {
        len = static_cast<int>(s.size());
    }

    Palindrome(const Palindrome&) = delete;
    Palindrome(Palindrome&&) = delete;
    Palindrome& operator=(const Palindrome&) = delete;
    Palindrome& operator=(Palindrome&&) = delete;

    bool IsPalindromicString(Method m = Method::Manacher) {
        return IsPalindromicString(0, len - 1, m);
    }

    bool IsPalindromicString(int left, int right, Method m = Method::Manacher) {
        bool res = true;
        switch (m) {
            case Method::Manacher: {
                break;
            }

            case Method::DoublePointer: {
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

            case Method::DP: {
                DPPreprocess();
                res = dp[left][right];
                break;
            }

            default:
                break;
        }

        return res;
    }

    int PalindromicSubstringNum(const std::string& s) {
        //
    }

private:
    void DPPreprocess() {
        dp.resize(len, std::vector<bool>(len));
        for (int i = len - 1; i >= 0; --i) {
            for (int j = i; j < len; ++j) {
                dp[i][j] = s[i] == s[j] && (j - i < 3 || dp[i + 1][j - 1]);
            }
        }
    }

    const std::string& s;
    int len;
    std::vector<std::vector<bool>> dp;
};