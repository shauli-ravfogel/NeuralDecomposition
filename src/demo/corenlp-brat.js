function annotate() {
  text = document.getElementById("text").value;
  preposition = document.getElementById("preposition").value;
  num_substitution = document.getElementById("num_substitution").value;
  text_based = document.getElementById("text_based").checked;
  fetch('http://nlp.biu.ac.il/~lazary/syntax_extractor/?text=' + text)
    .then(function(response) {
      return response.text()
    }).then(function(body) {
      // document.body.innerHTML = body
      // alert(body)
      out = document.getElementById("out-text");
      out.innerHTML = body;
      // out.type = 'text';
    })
};

