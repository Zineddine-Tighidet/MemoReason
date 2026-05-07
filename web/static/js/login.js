function loginApp(initialError = '') {
    return {
        username: '',
        password: '',
        error: String(initialError || ''),
        loading: false,

        login(event) {
            this.error = '';
            this.loading = true;
            const form = event && event.target ? event.target : document.querySelector('form[action="/login"]');
            if (form && typeof form.submit === 'function') {
                form.submit();
                return;
            }
            window.location.href = '/login';
        },
    };
}

window.loginApp = loginApp;
