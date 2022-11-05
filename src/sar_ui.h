#ifndef _SAR_UI_H_
#define _SAR_UI_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "common.h"

#include <FL/Fl.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Menu_Item.H>
#include <FL/Fl_Widget.H>
#include <FL/fl_draw.H>
// The X.h header that is ultimately included via FLTK dependencies
// defines Success as a macro with value 0, which conflicts with
// other uses of Success. We do not use the X.h macro explicitly,
// or any other macro named Success, so undefine it here.
#undef Success

// Forward declaration
class SarImageWidget;

class SarUI {
  public:
    SarUI(int width, int height);
    ~SarUI();
    int GetWidth() { return m_width; }
    int GetHeight() { return m_height; }

    SarGpuKernel GetSelectedKernel() {
        std::lock_guard<std::mutex> guard(m_lock);
        return m_kernel;
    }

    void SetSelectedKernel(SarGpuKernel kernel) {
        m_lock.lock();
        m_kernel = kernel;
        m_lock.unlock();
    }

    void UpdateImage(const uint32_t *pix, int width, int height);
    void StartEventLoop();
    void Close();
    bool IsStopping();
    std::pair<int, int> GetImageScreenDimensions();

    void KernelSelectionCallback(Fl_Widget *widget);

  private:
    int m_width;
    int m_height;
    static const int m_menu_height{30};
    SarGpuKernel m_kernel{SarGpuKernel::DoublePrecision};
    int m_kernel_ref_ind{0};
    bool m_stop{false};
    std::mutex m_lock;

    std::unique_ptr<Fl_Menu_Bar> m_file_menu;
    std::unique_ptr<Fl_Choice> m_kernel_menu;
    std::unique_ptr<Fl_Group> m_menu_group;
    std::unique_ptr<SarImageWidget> m_sar_image;
    std::unique_ptr<Fl_Double_Window> m_window;
};

#endif // _SAR_UI_H_