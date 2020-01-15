class Social extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.shadowRoot.innerHTML = `
        <style>
        .social ul {
            list-style-type: none;
            margin: 0;
          }
          .social ul:after {
            content: "";
            display: block;
            clear: both;
          }
          .social li {
            float: left;
          }
          .social a {
            border: 1px solid #ffffff;
            border-radius: 50%;
            display: block;
            font-size: 24px;
            height: 50px;
            width: 50px;
            text-align: center;
            line-height: 52px;
            margin-left: 10px;
            color: #fff;
          }
          
        </style>
        <div class="social">
        <ul>
            <li>
                <a href="https://www.linkedin.com/company/netherlands-escience-center" class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                        <path fill="#fff"
                            d="M4.98 3.5c0 1.381-1.11 2.5-2.48 2.5s-2.48-1.119-2.48-2.5c0-1.38 1.11-2.5 2.48-2.5s2.48 1.12 2.48 2.5zm.02 4.5h-5v16h5v-16zm7.982 0h-4.968v16h4.969v-8.399c0-4.67 6.029-5.052 6.029 0v8.399h4.988v-10.131c0-7.88-8.922-7.593-11.018-3.714v-2.155z" />
                    </svg>
                </a>
            </li>
            <li>
                <a href="https://twitter.com/esciencecenter" class="icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">
                        <path fill="#fff"
                            d="M24 4.557c-.883.392-1.832.656-2.828.775 1.017-.609 1.798-1.574 2.165-2.724-.951.564-2.005.974-3.127 1.195-.897-.957-2.178-1.555-3.594-1.555-3.179 0-5.515 2.966-4.797 6.045-4.091-.205-7.719-2.165-10.148-5.144-1.29 2.213-.669 5.108 1.523 6.574-.806-.026-1.566-.247-2.229-.616-.054 2.281 1.581 4.415 3.949 4.89-.693.188-1.452.232-2.224.084.626 1.956 2.444 3.379 4.6 3.419-2.07 1.623-4.678 2.348-7.29 2.04 2.179 1.397 4.768 2.212 7.548 2.212 9.142 0 14.307-7.721 13.995-14.646.962-.695 1.797-1.562 2.457-2.549z" />
                    </svg>
                </a>
            </li>
            <li>
                <a href="https://www.youtube.com/c/netherlandseScienceCenter" target="_blank"
                    class="icon-svg">
                    <svg version="1.1" xmlns="http://www.w3.org/2000/svg"
                        xmlns:xlink="http://www.w3.org/1999/xlink" x="0px" y="0px" width="27px"
                        height="20px" viewBox="-0.8 86.793 27 20"
                        enable-background="new -0.8 86.793 27 20" xml:space="preserve">
                        <path fill="#fff"
                            d="M24.943,101.787c0,0-0.244,1.729-0.995,2.484c-0.952,0.996-2.013,1.002-2.501,1.057c-3.498,0.258-8.747,0.258-8.747,0.258
                        h-0.013c0,0-5.246,0-8.743-0.258c-0.488-0.055-1.553-0.061-2.505-1.057c-0.751-0.756-0.989-2.484-0.989-2.484s-0.25-2.02-0.25-4.045
                        V95.85c0-2.027,0.25-4.053,0.25-4.053s0.244-1.721,0.989-2.479c0.952-0.994,2.203-0.965,2.759-1.066C6.2,88.061,12.7,88,12.7,88
                        s5.256,0.012,8.753,0.256c0.488,0.062,1.55,0.068,2.502,1.062c0.75,0.758,0.994,2.479,0.994,2.479S25.2,93.822,25.2,95.85v1.893
                        C25.193,99.762,24.943,101.787,24.943,101.787L24.943,101.787z M10.112,93.549v7.025l6.751-3.529L10.112,93.549L10.112,93.549z">
                        </path>
                    </svg>
                </a>
            </li>
        </ul>
    </div>
</div>
      `;
    }

}

customElements.define('nlesc-social', Social);
