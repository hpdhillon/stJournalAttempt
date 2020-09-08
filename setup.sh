mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"hd227@scarletmail.rutgers.edu"\"\n\
" > ~/.streamlit/credentials.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
