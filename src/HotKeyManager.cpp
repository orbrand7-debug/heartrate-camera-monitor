#pragma once
#include <windows.h>
#include <string>
#include <vector>
#include <sstream>
#include <map>

/**
 * @class HotKeyManager
 * @brief Detects keyboard combinations globally using Win32.
 */
class HotKeyManager {
public:
    static bool is_pressed(const std::string& str) {
        static std::map<std::string, int> vkm = {
            {"CTRL", VK_CONTROL}, {"ALT", VK_MENU}, {"SHIFT", VK_SHIFT}, {"WIN", VK_LWIN},
            {"SPACE", VK_SPACE}, {"ESC", VK_ESCAPE}, {"INS", VK_INSERT}, {"DEL", VK_DELETE},
            {"HOME", VK_HOME}, {"END", VK_END}, {"PGUP", VK_PRIOR}, {"PGDN", VK_NEXT},
            {"UP", VK_UP}, {"DOWN", VK_DOWN}, {"LEFT", VK_LEFT}, {"RIGHT", VK_RIGHT}
        };

        std::stringstream ss(str);
        std::string segment;
        while (std::getline(ss, segment, '+')) {
            for (auto& c : segment) c = (char)toupper(c);
            int vk = vkm.contains(segment) ? vkm[segment] : (segment.size() == 1 ? segment[0] : 0);
            if (vk == 0 || !(GetAsyncKeyState(vk) & 0x8000)) return false;
        }
        return true;
    }
};