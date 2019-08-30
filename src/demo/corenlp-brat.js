function annotate(sentence_based) {
  let text = document.getElementById("text").value;
  // if (sentence_based) {
  //   text = document.getElementById("text_sentence").value;
  // } else {
  //   text = document.getElementById("text_word").value;
  // }

  fetch('http://nlp.biu.ac.il/~lazary/syntax_extractor/?text=' + text
      + '&sentence_based=' + sentence_based)
    .then(function(response) {
      return response.text()
    }).then(function(body) {
      // document.body.innerHTML = body
      // alert(body)
    let ans = JSON.parse(body);
    let out = document.getElementById("out-text");
      out.innerHTML = ans['syntax'];

      out = document.getElementById("out-text-baseline");
      out.innerHTML = ans['baseline'];
      // out.type = 'text';
    })
}

