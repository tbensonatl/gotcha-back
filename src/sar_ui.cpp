#include "sar_ui.h"

#include "helpers.h"

class SarImageWidget : public Fl_Widget {
  public:
    SarImageWidget() = delete;
    SarImageWidget(int x, int y, int w, int h, const char *label = 0);
    virtual ~SarImageWidget() {}
    void UpdateImage(const uint32_t *image, int width, int height);
    std::pair<int, int> GetImageScreenDimensions();

    void resize(int x, int y, int w, int h);

  protected:
    void draw();

  private:
    static const int NUM_WINDOW_BUFFERS = 2;
    int m_native_width;
    int m_native_height;
    int m_screen_width;
    int m_screen_height;
    int m_height_offset;
    int m_active_window_ind{0};
    std::vector<uint32_t> m_native_image;
    std::vector<uint32_t> m_resized_image[NUM_WINDOW_BUFFERS];
    // The active lock covers m_active_window_ind, m_screen_width, and
    // m_screen_height. These values will typically be updated once a new image
    // has been written into an inactive buffer.
    std::mutex m_active_lock;
    // The pending lock covers updating the native image and inactive
    // resampled image.
    std::mutex m_pending_lock;

    void Resample(int screen_width = -1, int screen_height = -1);
};

SarImageWidget::SarImageWidget(int x, int y, int w, int h, const char *label)
    : Fl_Widget(x, y, w, h, label), m_native_width(w), m_native_height(h),
      m_screen_width(w), m_screen_height(h), m_height_offset(y) {
    m_native_image.resize(m_native_width * m_native_height);
    memset(m_native_image.data(), 0,
           sizeof(uint32_t) * m_native_width * m_native_height);
    for (int i = 0; i < NUM_WINDOW_BUFFERS; i++) {
        m_resized_image[i].resize(m_screen_width * m_screen_height);
        memset(m_resized_image[i].data(), 0,
               sizeof(uint32_t) * m_screen_width * m_screen_height);
    }
}

void SarImageWidget::draw() {
    m_active_lock.lock();
    fl_draw_image(reinterpret_cast<uint8_t *>(
                      m_resized_image[m_active_window_ind].data()),
                  0, m_height_offset, m_screen_width, m_screen_height, 4, 0);
    m_active_lock.unlock();
}

void SarImageWidget::resize(int x, int y, int w, int h) {
    Fl_Widget::resize(x, y, w, h);
    Resample(w, h);
}

// Typically, the SAR image will have been resampled to the current
// window size on the GPU being being pushed to the SarUI class. However,
// when the window is resized, it may take 1-2 reconstructions before
// the resampling will be handled on the GPU. This resampler is a fallback
// for that interim period. If the native image size matches the screen
// size, it's just a memory copy, but if not, then the native image
// is resampled via nearest neighbor interpolation to fill the window.
// The resampling that is done on the GPU is currently bilinear interpolation
// and thus produces better quality images (at least in the case that the
// window is larger than the native reconstruction).
void SarImageWidget::Resample(int screen_width, int screen_height) {
    m_pending_lock.lock();
    screen_width = (screen_width >= 0) ? screen_width : m_screen_width;
    screen_height = (screen_height >= 0) ? screen_height : m_screen_height;
    const int next_resample_ind =
        (m_active_window_ind + 1) % NUM_WINDOW_BUFFERS;
    std::vector<uint32_t> &img = m_resized_image[next_resample_ind];
    img.resize(screen_width * screen_height);
    if (screen_width == m_native_width && screen_height == m_native_height) {
        memcpy(img.data(), m_native_image.data(),
               sizeof(uint32_t) * m_native_width * m_native_height);
    } else {
        std::vector<int> x_native_coord(screen_width);
        const float width_conv_factor =
            static_cast<double>(m_native_width) / screen_width;
        const float height_conv_factor =
            static_cast<double>(m_native_height) / screen_height;
        for (int i = 0; i < screen_width; i++) {
            x_native_coord[i] = static_cast<int>(roundf(i * width_conv_factor));
        }
        for (int i = 0; i < screen_height; i++) {
            const int iy = static_cast<int>(roundf(i * height_conv_factor));
            for (int j = 0; j < screen_width; j++) {
                const int ix = x_native_coord[j];
                img[i * screen_width + j] =
                    m_native_image[iy * m_native_width + ix];
            }
        }
    }
    m_active_lock.lock();
    m_active_window_ind = (m_active_window_ind + 1) % NUM_WINDOW_BUFFERS;
    m_screen_width = screen_width;
    m_screen_height = screen_height;
    m_active_lock.unlock();
    m_pending_lock.unlock();
    redraw();
    Fl::awake();
}

void SarImageWidget::UpdateImage(const uint32_t *image, int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }
    m_pending_lock.lock();
    if (static_cast<size_t>(width * height) > m_native_image.size()) {
        m_native_image.resize(width * height);
    }
    memcpy(m_native_image.data(), image, sizeof(uint32_t) * width * height);
    m_native_width = width;
    m_native_height = height;
    m_pending_lock.unlock();
    Resample();
}

std::pair<int, int> SarImageWidget::GetImageScreenDimensions() {
    m_active_lock.lock();
    const int w = m_screen_width;
    const int h = m_screen_height;
    m_active_lock.unlock();
    return std::make_pair(w, h);
}

static void kernel_cb(Fl_Widget *widget, void *userdata) {
    SarUI *ui = reinterpret_cast<SarUI *>(userdata);
    ui->KernelSelectionCallback(widget);
}

void SarUI::KernelSelectionCallback(Fl_Widget *widget) {
    Fl_Choice *choice = reinterpret_cast<Fl_Choice *>(widget);
    const int kernel_val = (choice->value() - m_kernel_ref_ind) + 1;
    switch (kernel_val) {
    case static_cast<int>(SarGpuKernel::DoublePrecision):
        SetSelectedKernel(SarGpuKernel::DoublePrecision);
        break;
    case static_cast<uintptr_t>(SarGpuKernel::MixedPrecision):
        SetSelectedKernel(SarGpuKernel::MixedPrecision);
        break;
    case static_cast<uintptr_t>(SarGpuKernel::IncrPhaseLookup):
        SetSelectedKernel(SarGpuKernel::IncrPhaseLookup);
        break;
    case static_cast<uintptr_t>(SarGpuKernel::NewtonRaphsonTwoIter):
        SetSelectedKernel(SarGpuKernel::NewtonRaphsonTwoIter);
        break;
    case static_cast<uintptr_t>(SarGpuKernel::IncrRangeSmem):
        SetSelectedKernel(SarGpuKernel::IncrRangeSmem);
        break;
    case static_cast<uintptr_t>(SarGpuKernel::SinglePrecision):
        SetSelectedKernel(SarGpuKernel::SinglePrecision);
        break;
    default:
        LOG("Invalid kernel selection %d", kernel_val);
    }
}

static void exit_cb(Fl_Widget *, void *userdata) {
    SarUI *ui = reinterpret_cast<SarUI *>(userdata);
    ui->Close();
}

void SarUI::Close() {
    m_lock.lock();
    m_stop = true;
    m_lock.unlock();
    Fl::awake();
}
SarUI::SarUI(int width, int height) : m_width(width), m_height(height) {
    m_window.reset(
        new Fl_Double_Window(m_width, m_height + m_menu_height, "Video SAR"));
    const int file_menu_width = 50;
    const int kernel_menu_offset = 50;
    const int kernel_menu_width = 200;
    m_menu_group.reset(new Fl_Group(0, 0, m_width, m_menu_height));
    m_file_menu.reset(new Fl_Menu_Bar(0, 0, file_menu_width, m_menu_height));
    m_file_menu->add("&File/E&xit", FL_COMMAND + 'x', exit_cb, this);
    m_file_menu->box(FL_NO_BOX);
    m_kernel_menu.reset(new Fl_Choice(file_menu_width + kernel_menu_offset, 0,
                                      kernel_menu_width, m_menu_height,
                                      "&Kernel"));
    m_kernel_ref_ind =
        m_kernel_menu->add("FP64 (Reference)", 0, kernel_cb, this);
    m_kernel_menu->add("Mixed Precision (Opt1)", 0, kernel_cb, this);
    m_kernel_menu->add("IncrPhaseLookup (Opt2)", 0, kernel_cb, this);
    m_kernel_menu->add("Newton-Raphson 2 Iter (Opt3)", 0, kernel_cb, this);
    m_kernel_menu->add("IncrRangeSmem (Opt4)", 0, kernel_cb, this);
    m_kernel_menu->add("SinglePrecision (Opt5)", 0, kernel_cb, this);
    m_kernel_menu->value(m_kernel_ref_ind);
    m_menu_group->resizable(nullptr);
    m_menu_group->end();
    m_sar_image.reset(new SarImageWidget(0, m_menu_height, m_width, m_height));
    m_window->resizable(m_sar_image.get());
    m_window->size_range(width / 2, height / 2, 2048, 2048);
    m_window->end();
}

SarUI::~SarUI() {
    m_file_menu.reset(nullptr);
    m_kernel_menu.reset(nullptr);
    m_menu_group.reset(nullptr);
    m_sar_image.reset(nullptr);
    m_window.reset(nullptr);
}

bool SarUI::IsStopping() {
    std::lock_guard guard(m_lock);
    return m_stop;
}

void SarUI::UpdateImage(const uint32_t *pix, int width, int height) {
    if (IsStopping()) {
        return;
    }
    m_sar_image->UpdateImage(pix, width, height);
}

std::pair<int, int> SarUI::GetImageScreenDimensions() {
    return (m_sar_image) ? m_sar_image->GetImageScreenDimensions()
                         : std::make_pair(0, 0);
}

void SarUI::StartEventLoop() {
    Fl::visual(FL_RGB);
    m_window->show(0, nullptr);
    Fl::lock();
    while (Fl::first_window()) {
        if (IsStopping()) {
            m_window->hide();
            break;
        }
        Fl::wait();
    }
}