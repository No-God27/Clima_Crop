let clima= document.getElementById('clima');
let wheat = document.getElementById('wheat');
// let left= document.getElementById('left');
// let right = document.getElementById('right');
let scrollHeight = document.body.scrollHeight - window.innerHeight;

window.addEventListener('scroll', () => {
    let value = window.scrollY;
 if (value <= scrollHeight) {
    clima.style.marginTop = value * 1.5 + 'px';
    wheat.style.marginTop = value * .25+ 'px';

  }

});

let sections = document.querySelectorAll('section');
let navLinks = document.querySelectorAll('header nav a')

window.onscroll = () =>
{
  sections.forEach(sec =>
  {
    let top = window.scrollY
    let offset = sec.offsetTop - 150;
    let height = sec.offsetHeight;
    let id =sec.getAttribute('id')

      if(top>=offset && top < offset + height)
      {navLinks.forEach(links =>
      {
        links.classList.remove('active');
        document.querySelector('header nav a[href*=' + id + ']').classList.add('active');

      })

    };
  })
};

function openContactDialog() {
  document.getElementById("contactOverlay").style.display = "flex";
}

function closeContactDialog() {
  document.getElementById("contactOverlay").style.display = "none";
};