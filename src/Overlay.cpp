/**
 * @file Overlay.cpp
 * @brief Implementation of the transparent Win32 HUD with global hotkey support.
 */

#include "Overlay.hpp"
#include <stdexcept>
#include <print>
#include <format>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>

struct GDIObjectDeleter {
    void operator()(HGDIOBJ obj) const { if (obj) DeleteObject(obj); }
};
using UniqueGDIObject = std::unique_ptr<std::remove_pointer_t<HGDIOBJ>, GDIObjectDeleter>;


/**
 * @brief Constructor for the HUD Overlay.
 * Initializes the Win32 window and registers a global system hotkey.
 */
Overlay::Overlay(const AppConfig& c) 
    : m_hInstance(GetModuleHandle(NULL)), m_cfg(c) {
    const char* CLASS_NAME = "HeartrateHUDClass";

    WNDCLASS wc = {};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = m_hInstance;
    wc.lpszClassName = CLASS_NAME;
    // Pure black background used as the transparency colorkey
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);

    if (!RegisterClass(&wc)) {
        throw std::runtime_error("Failed to register Win32 window class.");
    }

    // Create a topmost, transparent, click-through layered window
    m_hwnd = CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE,
        CLASS_NAME, "Heartbeat HUD",
        WS_POPUP,
        m_cfg.hud.x, m_cfg.hud.y, 
        m_cfg.hud.width, m_cfg.hud.height,
        NULL, NULL, m_hInstance, this
    );

    if (!m_hwnd) {
        throw std::runtime_error("Failed to create HUD window.");
    }
    m_window_w = m_cfg.hud.width;
    m_window_h = m_cfg.hud.height;

    // Configure transparency: Black pixels are invisible, global alpha controls opacity
    SetLayeredWindowAttributes(m_hwnd, RGB(0, 0, 0), m_cfg.hud.alpha, LWA_COLORKEY | LWA_ALPHA);

    // Register the Global HotKey (Event-driven, does not miss presses)
    UINT modifiers = 0, vk = 0;
    parse_hotkey(m_cfg.hud.hotkey_toggle_debug, modifiers, vk);
    
    // Attempt registration. Note: RegisterHotKey will fail if out_vk is 0.
    if (vk == 0 || !RegisterHotKey(m_hwnd, HOTKEY_ID, modifiers, vk)) {
        std::println(stderr, "Hotkey Error: '{}' is invalid or already in use.", 
                     m_cfg.hud.hotkey_toggle_debug);
    }

    ShowWindow(m_hwnd, SW_SHOW);
}

/**
 * @brief Destructor. Unregisters the system hotkey.
 */
Overlay::~Overlay() {
    stop();
    UnregisterHotKey(m_hwnd, HOTKEY_ID);
    if (m_hwnd) DestroyWindow(m_hwnd);
}

void Overlay::update_bpm(double bpm) {
    m_bpm = bpm;
    // Request a repaint on the UI thread
    if (m_hwnd) InvalidateRect(m_hwnd, NULL, FALSE);
}

void Overlay::update_frame(const cv::Mat& frame) {
    if (frame.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        // Copy the frame to our internal buffer for the UI thread to use
        frame.copyTo(m_frame);
        m_frame_w = frame.cols;
        m_frame_h = frame.rows;
    }
    if (m_hwnd && m_frame_w > 0 && m_frame_h > 0) {
        const int max_w = m_cfg.hud.width;
        const int max_h = m_cfg.hud.height;
        const double scale = std::min(static_cast<double>(max_w) / m_frame_w,
                                      static_cast<double>(max_h) / m_frame_h);
        const int new_w = std::max(1, static_cast<int>(std::lround(m_frame_w * scale)));
        const int new_h = std::max(1, static_cast<int>(std::lround(m_frame_h * scale)));
        if (new_w != m_window_w || new_h != m_window_h) {
            m_window_w = new_w;
            m_window_h = new_h;
            SetWindowPos(m_hwnd, NULL, 0, 0, m_window_w, m_window_h,
                         SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }
    }
    if (m_hwnd) InvalidateRect(m_hwnd, NULL, FALSE);
}

void Overlay::stop() {
    m_running = false;
    if (m_hwnd) PostMessage(m_hwnd, WM_CLOSE, 0, 0);
}

void Overlay::run() {
    MSG msg = {};
    while (m_running && GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

/**
 * @brief Translates a configuration string into Win32 HotKey modifiers and Virtual Key codes.
 * @param str The hotkey string from config (e.g., "Ctrl+Shift+F1").
 * @param out_mod [out] The Win32 modifier bitmask (MOD_CONTROL, etc.).
 * @param out_vk [out] The Win32 Virtual-Key code.
 */
void Overlay::parse_hotkey(const std::string& str, UINT& out_mod, UINT& out_vk) {
    // 1. Define the mapping for Virtual Keys (VK)
    static const std::map<std::string, int> key_map = {
        {"SPACE", VK_SPACE}, {"ESC", VK_ESCAPE}, {"INS", VK_INSERT}, {"DEL", VK_DELETE},
        {"HOME", VK_HOME}, {"END", VK_END}, {"PGUP", VK_PRIOR}, {"PGDN", VK_NEXT},
        {"UP", VK_UP}, {"DOWN", VK_DOWN}, {"LEFT", VK_LEFT}, {"RIGHT", VK_RIGHT},
        {"F1", VK_F1}, {"F2", VK_F2}, {"F3", VK_F3}, {"F4", VK_F4}, {"F5", VK_F5},
        {"F6", VK_F6}, {"F7", VK_F7}, {"F8", VK_F8}, {"F9", VK_F9}, {"F10", VK_F10},
        {"F11", VK_F11}, {"F12", VK_F12}
    };

    out_mod = MOD_NOREPEAT; // Standard behavior: don't auto-repeat toggle
    out_vk = 0;

    std::stringstream ss(str);
    std::string segment;
    
    while (std::getline(ss, segment, '+')) {
        // Normalize to uppercase
        for (auto& c : segment) c = (char)toupper(static_cast<unsigned char>(c));

        // 2. Check for Modifiers (Special Win32 flags for RegisterHotKey)
        if (segment == "CTRL") {
            out_mod |= MOD_CONTROL;
        } else if (segment == "ALT") {
            out_mod |= MOD_ALT;
        } else if (segment == "SHIFT") {
            out_mod |= MOD_SHIFT;
        } else if (segment == "WIN") {
            out_mod |= MOD_WIN;
        } 
        // 3. Check for specific named keys (F1, Home, Space, etc.)
        else if (key_map.contains(segment)) {
            out_vk = key_map.at(segment);
        }
        // 4. Check for single character keys (A-Z, 0-9)
        else if (segment.size() == 1) {
            out_vk = static_cast<UINT>(segment[0]);
        }
    }
}

/**
 * @brief Draws the OpenCV frame and BPM text using GDI.
 */
void Overlay::paint(HDC hdc) {
    RECT rect;
    GetClientRect(m_hwnd, &rect);
    int hud_w = rect.right - rect.left;
    int hud_h = rect.bottom - rect.top;

    // 1. Render the Camera Frame (if available)
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (!m_frame.empty()) {
            // GDI works best with 32-bit BGRA
            cv::Mat bgra;
            cv::cvtColor(m_frame, bgra, cv::COLOR_BGR2BGRA);

            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = bgra.cols;
            bmi.bmiHeader.biHeight = -bgra.rows; // Negative for top-down orientation
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            SetStretchBltMode(hdc, COLORONCOLOR);
            StretchDIBits(hdc, 
                0, 0, hud_w, hud_h,           // Destination (Scaled)
                0, 0, bgra.cols, bgra.rows,    // Source
                bgra.data, &bmi, DIB_RGB_COLORS, SRCCOPY);
        }
    }

    // 2. Render Text Overlay
    UniqueGDIObject hFont(CreateFontA(
        m_cfg.hud.font_size, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, 
        DEFAULT_CHARSET, OUT_OUTLINE_PRECIS, CLIP_DEFAULT_PRECIS, 
        CLEARTYPE_QUALITY, VARIABLE_PITCH, 
        m_cfg.hud.font_name.c_str()
    ));
    
    HGDIOBJ hOldFont = SelectObject(hdc, hFont.get());
    SetBkMode(hdc, TRANSPARENT);

    std::string text = m_bpm > 0 
        ? std::format("BPM: {:.1f}", m_bpm.load()) 
        : "Analyzing...";

    // Draw shadow for readability
    SetTextColor(hdc, RGB(0, 0, 0));
    TextOutA(hdc, 2, 2, text.c_str(), (int)text.length());
    
    // Draw foreground
    SetTextColor(hdc, RGB(m_cfg.hud.r, m_cfg.hud.g, m_cfg.hud.b));
    TextOutA(hdc, 0, 0, text.c_str(), (int)text.length());

    SelectObject(hdc, hOldFont);
}

/**
 * @brief Static Win32 Procedure. Routes messages to the class instance.
 */
LRESULT CALLBACK Overlay::WindowProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    Overlay* pOverlay = nullptr;
    if (m == WM_NCCREATE) {
        CREATESTRUCT* pCreate = reinterpret_cast<CREATESTRUCT*>(l);
        pOverlay = reinterpret_cast<Overlay*>(pCreate->lpCreateParams);
        SetWindowLongPtr(h, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pOverlay));
    } else {
        pOverlay = reinterpret_cast<Overlay*>(GetWindowLongPtr(h, GWLP_USERDATA));
    }

    if (pOverlay) {
        switch (m) {
            case WM_HOTKEY:
                if (w == pOverlay->HOTKEY_ID) {
                    pOverlay->m_debug_enabled = !pOverlay->m_debug_enabled;
                }
                return 0;

            case WM_PAINT: {
                PAINTSTRUCT ps;
                HDC hdc = BeginPaint(h, &ps);
                pOverlay->paint(hdc);
                EndPaint(h, &ps);
                return 0;
            }

            case WM_DESTROY: {
                PostQuitMessage(0);
                return 0;
            }
        }
    }
    return DefWindowProc(h, m, w, l);
}
