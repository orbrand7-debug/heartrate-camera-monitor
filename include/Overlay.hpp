#pragma once
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <atomic>
#include <mutex>
#include <string>
#include "Config.hpp"

/**
 * @class Overlay
 * @brief Managed Win32 HUD with event-driven global hotkey handling.
 */
class Overlay {
public:
    /**
     * @brief Creates the HUD and registers the global hotkey.
     * @param c Application configuration.
     */
    explicit Overlay(const AppConfig& c);
    
    /**
     * @brief Unregisters hotkeys and cleans up resources.
     */
    ~Overlay();

    /**
     * @brief Runs the Win32 message loop.
     */
    void run();
    
    /**
     * @brief Signals the thread to stop.
     */
    void stop();

    /**
     * @brief Updates the numerical BPM display.
     */
    void update_bpm(double b);

    /**
     * @brief Thread-safe update of the display frame.
     */
    void update_frame(const cv::Mat& f);

    /**
     * @brief Returns whether debug mode is currently toggled on.
     */
    bool is_debug_mode() const { return m_debug_enabled; }

private:
    static LRESULT CALLBACK WindowProc(HWND h, UINT m, WPARAM w, LPARAM l);
    void paint(HDC hdc);
    
    /**
     * @brief Translates config string (e.g., "Ctrl+Alt+D") into Win32 HotKey flags.
     * @param str The hotkey string.
     * @param out_mod Bitwise Win32 modifiers (MOD_CONTROL, etc).
     * @param out_vk The Virtual Key code.
     */
    void parse_hotkey(const std::string& str, UINT& out_mod, UINT& out_vk);

    std::atomic<bool> m_running{true};
    std::atomic<bool> m_debug_enabled{false};
    std::atomic<double> m_bpm{0.0};
    
    std::mutex m_mtx;
    cv::Mat m_frame;
    
    HWND m_hwnd{nullptr};
    HINSTANCE m_hInstance;
    AppConfig m_cfg;
    const int HOTKEY_ID = 101; // Unique ID for this app's hotkey
};