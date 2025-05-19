document.addEventListener("DOMContentLoaded", () => {
    // Handle form submission for upload and search
    const forms = document.querySelectorAll("form")
    const progressBar = document.getElementById("upload-progress")
  
    forms.forEach((form) => {
      form.addEventListener("submit", function (e) {
        const fileInput = this.querySelector('input[type="file"]')
  
        if (fileInput && fileInput.files.length > 0) {
          // Show progress bar
          if (progressBar) {
            progressBar.classList.remove("d-none")
            const progressBarInner = progressBar.querySelector(".progress-bar")
            progressBarInner.style.width = "0%"
            progressBarInner.textContent = "0%"
  
            // Simulate upload progress (since we can't track actual progress without AJAX)
            let progress = 0
            const interval = setInterval(() => {
              progress += 5
              if (progress > 90) {
                clearInterval(interval)
              }
              progressBarInner.style.width = progress + "%"
              progressBarInner.textContent = progress + "%"
            }, 500)
          }
  
          // Disable submit button
          const submitBtn = this.querySelector('button[type="submit"]')
          if (submitBtn) {
            submitBtn.disabled = true
            submitBtn.textContent = submitBtn.id === "search-btn" ? "Đang tìm kiếm..." : "Đang tải lên..."
          }
        }
      })
    })
  })
  