import React from 'react';
import './CSS/Navbar.css';

const Navbar = ({ handleChange }) => {
  return (
    <nav className="navbar">
      <div className="navbar-left">
        <h1 className="navbar-title">CryptoIQ</h1>
      </div>
      <div className="navbar-right">
        <form action="">
          <input type="text" className="navbar-input" placeholder="Search coins" onChange={handleChange}/>
        </form>
      </div>
    </nav>
  );
}

export default Navbar;
